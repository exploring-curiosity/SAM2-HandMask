import cv2
import numpy as np

# Load video and extract frame 0
video_path = 'Reach.mp4'
cap = cv2.VideoCapture(video_path)
success, frame_0 = cap.read()
cap.release()

if not success:
    print("Failed to extract frame 0")
    exit()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getUnconnectedOutLayersNames()

# Prepare image
blob = cv2.dnn.blobFromImage(frame_0, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Run forward pass
outputs = net.forward(layer_names)

# Process detections
boxes = []
confidences = []
class_ids = []
height, width = frame_0.shape[:2]

# Filter for bowl and wine glass
target_classes = ['bowl', 'wine glass', 'cup']
target_class_ids = [classes.index(cls) for cls in target_classes]

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and class_id in target_class_ids:
            center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype('int')
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes and labels
colors = np.random.uniform(0, 255, size=(len(target_classes), 3))

for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    color = colors[target_class_ids.index(class_ids[i])]
    cv2.rectangle(frame_0, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame_0, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Print coordinates
    center_x = x + w // 2
    center_y = y + h // 2
    print(f"{classes[class_ids[i]]}: Center coordinates (x, y) = ({center_x}, {center_y})")

# Display the result
cv2.imshow("Bowl and Cup Detection", frame_0)
cv2.waitKey(0)
cv2.destroyAllWindows()
