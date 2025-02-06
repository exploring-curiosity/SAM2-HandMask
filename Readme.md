# SAM2 Hand Masking Project

This project demonstrates how to use the SAM2 (Segment Anything Model 2) for masking hands in videos. It provides both interactive and script-based implementations for generating masked videos.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [SAM2 Repository Setup](#sam2-repository-setup)
3. [Checkpoint Setup](#checkpoint-setup)
4. [Running the Masking Function](#running-the-masking-function)
5. [Output Files](#output-files)
6. [NOTE](#note)

---

## Environment Setup

### Prerequisites
- Python 3.10 or above is required.

### Steps
1. **Create a Python Virtual Environment**:
   ```bash
   python -m venv myenv

2. **Activate the Environment**:
    ```bash
    source myenv/bin/activate

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt

## SAM2 Repository Setup

1. **Clone the SAM2 Repository & navigate into it**:
    ```bash
    git clone https://github.com/facebookresearch/sam2.git && cd sam2

2. **Install the SAM2 Package**:
    ```bash
    pip install -e .

## Checkpoint Setup
1. **Navigate to the checkpoints folder**:
    ```bash
    cd checkpoints

2. **Download the required model checkpoints**:
    - Download script: 
        ```bash
        ./download_ckpts.sh
    - Alternatively, download the checkpoints manually from the [SAM2 GitHub page](https://github.com/facebookresearch/sam2?tab=readme-ov-file#getting-started) and place them in the checkpoints folder.

## Running the Masking Function

### Interactive Mode (Jupyter Notebook)
1. Move the `trackHands.ipynb` notebook to your working directory.
2. Open the notebook and modify the following variables if needed:
    - `video_path`: Path to the input video file.
    - `output_video_path`: Path to save the masked video.
3. Execute the notebook cells to generate the masked video.

### Script Mode (Python File)
1. Move the `trackHands.py` script to your working directory.
2. Modify the following variables in the script if needed:
    - `video_path`: Path to the input video file.
    - `output_video_path`: Path to save the masked video.
3. Run the script:
    ```bash
    python trackHands.py

### Input Video
Use the provided `test.mp4` file or replace it with your own video file.

## Output Files

The following output files will be generated:
    - `output_video_Large_model.mp4`: Masked video using the SAM2 large model.
    - `output_video_small_model.mp4`: Masked video using the SAM2 small model.

## NOTE
MPS Systems: The script uses `torch.mps.empty_cache()` to avoid errors on MPS-based systems. If you're using CUDA or CPU, comment out or modify this line accordingly.