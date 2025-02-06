import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import imageio


video_path = '../test.mp4'
output_video_path = 'output_test.mp4'

def extract_all_frames(video_path, output_dir):
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # save every frame
        frame_filename = os.path.join(output_dir, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")


def trackHands(video_path, output_video_path, sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt", model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)

    # read the first frame (frame 0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the video.")

    cap.release()
    cv2.destroyAllWindows()

    # convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # initialize lists to store finger tip coordinates for each hand
    hand1_finger_tips = []
    hand2_finger_tips = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # extract landmarks for each hand
            landmarks = hand_landmarks.landmark
            
            # indices for the finger tips
            # thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20
            finger_tip_indices = [4, 8, 12, 16, 20]
            
            # get the (x, y) coordinates of the finger tips
            finger_tips = []
            for idx in finger_tip_indices:
                x = landmarks[idx].x * frame.shape[1]  # scale x to image width
                y = landmarks[idx].y * frame.shape[0]  # scale y to image height
                finger_tips.append((x, y))
            
            # store the finger tips in the appropriate list
            if i == 0:
                hand1_finger_tips = finger_tips
            elif i == 1:
                hand2_finger_tips = finger_tips

    else:
        print("No hands detected in frame 0.")

    cap.release()

    # convert the finger tips to NumPy arrays
    hand1_finger_tips = np.array(hand1_finger_tips)
    hand2_finger_tips = np.array(hand2_finger_tips)

    # Print the coordinates
    print("Hand 1 Finger Tips Coordinates:")
    print(hand1_finger_tips)
    print("Hand 2 Finger Tips Coordinates:")
    print(hand2_finger_tips)

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = f"{video_name}_dir"
    extract_all_frames(video_path, video_dir)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # take a look the first video frame
    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

    # change to cuda or other for other device types to optimize performance
    torch.mps.empty_cache()
    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id_hand1 = 1  # give a unique id to each object we interact with (it can be any integers)
    ann_obj_id_hand2 = 2  # give a unique id to each object we interact with (it can be any integers)
    prompts = {}

    points = np.array(hand1_finger_tips, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1,1,1,1,1], np.int32)
    prompts[ann_obj_id_hand1] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_hand1,
        points=points,
        labels=labels,
    )

    points = np.array(hand2_finger_tips, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1,1,1,1,1], np.int32)
    prompts[ann_obj_id_hand2] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id_hand2,
        points=points,
        labels=labels,
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    output_video_path = "output_video.mp4"
    temp_frame_dir = "temp_frames"
    os.makedirs(temp_frame_dir, exist_ok=True)

    # List to store the paths of the generated frames
    frame_paths = []

    for out_frame_idx in range(0, len(frame_names)):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        
        # Save the frame with the mask applied
        frame_path = os.path.join(temp_frame_dir, f"frame_{out_frame_idx:04d}.png")
        plt.savefig(frame_path)
        plt.close()
        
        # Append the frame path to the list
        frame_paths.append(frame_path)

    # Compile the frames into a video
    with imageio.get_writer(output_video_path, fps=30) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)


    print(f"Video saved to {output_video_path}")



    if os.path.exists(temp_frame_dir):
        # Delete the directory and all its contents
        shutil.rmtree(temp_frame_dir)
        print(f"Directory '{temp_frame_dir}' and all its contents have been deleted.")
    else:
        print(f"Directory '{temp_frame_dir}' does not exist.")

    if os.path.exists(video_dir):
        # Delete the directory and all its contents
        shutil.rmtree(video_dir)
        print(f"Directory '{video_dir}' and all its contents have been deleted.")
    else:
        print(f"Directory '{video_dir}' does not exist.")


trackHands(video_path, output_video_path)