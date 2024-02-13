# Vehicle Detection and Counting System

![Alt Text](vehicle-entry-count-YOLO-NAS/vehicle_gif.gif)

This repository contains a Python script for detecting vehicles in a video stream, tracking their movement, and counting the number of vehicles entering and leaving a specified region. The system utilizes a pre-trained YOLO (You Only Look Once) model for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for object tracking.

## Requirements

- Python 3.x
- OpenCV (cv2)
- PyTorch
- super_gradients (custom module)
- SORT (Simple Online and Realtime Tracking)

## Usage

1. Install the required packages using pip:
    ```bash
    pip install opencv-python torch
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vehicle-detection.git
    ```

3. Navigate to the repository directory:
    ```bash
    cd vehicle-detection
    ```

4. Run the Python script:
    ```bash
    python vehicle_detection.py
    ```

## Description

The script performs the following tasks:
- Reads a video stream from a specified file (`VehiclesEnteringandLeaving.mp4`).
- Utilizes a pre-trained YOLO model (`yolo_nas_m`) for object detection, detecting various classes of objects including vehicles.
- Tracks the detected objects using the SORT algorithm to maintain their identities across frames.
- Counts the number of vehicles entering and leaving a specified region using predefined boundaries (`limitup` and `limitdown`).
- Displays the output frame with bounding boxes around detected vehicles, along with the count of vehicles entering and leaving.

## Acknowledgements

- This project is inspired by the work of others in the field of computer vision and object tracking.
- The YOLO model used in this project is provided by the `super_gradients` module.
- The SORT algorithm implementation is based on the work of Alex Bewley.
