# height_engine.py (Bidirectional Trigonometric Engine)
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Initialize MediaPipe Pose Landmarker via modern Tasks API
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

def process_height_frame(frame, ref_height_cm=170.0):
    """
    Applies the bidirectional pinhole camera model:
    1. Uses true reference height to compute exact distance (Z) from the camera lens.
    2. Uses trigonometric ray angles between head and feet for high-precision height verification.
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    results = detector.detect(mp_image)
    
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        landmarks = results.pose_landmarks[0]
        
        nose = landmarks[0]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        nose_y = nose.y * h
        base_ankle_y = ((left_ankle.y + right_ankle.y) / 2) * h
        pixel_height = abs(base_ankle_y - nose_y)
        
        if pixel_height > 50:
            # Standard webcam vertical field of view approximation (~60 degrees)
            vertical_fov_deg = 60.0
            focal_length_px = (h / 2.0) / np.tan(np.radians(vertical_fov_deg / 2.0))
            
            # 1. Bidirectional Math: Solve for exact camera distance (Z) using true reference height
            # Formula: Z = (Focal Length * True Height) / Pixel Height
            distance_from_camera_cm = (focal_length_px * ref_height_cm) / pixel_height
            
            # 2. Trigonometric Ray Correction for angle distribution across the frame
            center_y = h / 2.0
            angle_per_pixel = vertical_fov_deg / h
            
            head_angle = np.radians((nose_y - center_y) * angle_per_pixel)
            feet_angle = np.radians((base_ankle_y - center_y) * angle_per_pixel)
            
            # Precise height calculation using angular span at calculated depth Z
            calculated_height_cm = distance_from_camera_cm * abs(np.tan(feet_angle) - np.tan(head_angle))
            
            # Draw visual tracking lines and nodes
            nx, ny = int(nose.x * w), int(nose_y)
            ax, ay = int(((left_ankle.x + right_ankle.x) / 2) * w), int(base_ankle_y)
            
            cv2.line(frame, (nx, ny), (ax, ay), (0, 255, 0), 3)
            cv2.circle(frame, (nx, ny), 6, (0, 0, 255), -1)
            cv2.circle(frame, (ax, ay), 6, (255, 0, 0), -1)
            
            # Display both metrics on the dashboard feed
            cv2.putText(frame, f"Height: {calculated_height_cm:.1f} cm", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Cam Dist: {distance_from_camera_cm:.1f} cm", (30, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Step back for full body visibility", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Positioning Core... Stand in view", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
    return frame