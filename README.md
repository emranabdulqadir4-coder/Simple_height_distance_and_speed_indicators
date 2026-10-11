# Simple_height_distance_and_speed_indicator

A real-time computer vision project built with Python, OpenCV, YOLOv8, MediaPipe, and Flask. This app measures human height, object distance, and speed using a live webcam feed and displays the results in a browser-based dashboard.

---

## Features

- Live webcam streaming
- Height estimation using MediaPipe pose landmarks
- Distance measurement between detected objects
- Speed tracking using object keypoints
- Real-time visual overlays on the video feed
- Browser-based calibration controls via Flask

---

## Tech Stack

- Python
- Flask
- OpenCV
- Ultralytics YOLOv8
- MediaPipe
- NumPy

---

## Project Structure

```text
├── main.py                 # Flask app and video streaming logic
├── height_engine.py        # Height estimation logic
├── distance_engine.py      # Distance measurement logic
├── speed_engine.py         # Motion speed calculation logic
├── templates/
│   └── indicators.html     # Web dashboard UI
├── requirements.txt        # Python dependencies
├── pose_landmarker_lite.task
├── yolov8n.pt
├── yolov8n-pose.pt
├── README.md
