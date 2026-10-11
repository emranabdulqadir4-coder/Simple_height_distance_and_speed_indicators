# distance_engine.py
import cv2
import math
from ultralytics import YOLO

# Load the standard, ultra-lightweight object detection model
model = YOLO('yolov8n.pt') 

def calculate_centroid(box):
    """Calculates the center (x, y) coordinate of a bounding box."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def process_distance_frame(frame, real_width, cal_distance):
    """Processes the frame to calculate distance between two detected objects using web inputs."""
    results = model(frame, verbose=False)
    detected_objects = []
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  
        for box in boxes:
            centroid = calculate_centroid(box)
            detected_objects.append(centroid)
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            
            if len(detected_objects) == 2:
                break
        if len(detected_objects) == 2:
            break
                
    # If exactly two objects are in the view, calculate distance
    if len(detected_objects) == 2:
        pt1 = detected_objects[0]
        pt2 = detected_objects[1]
        
        pixel_distance = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        
        # Derive a scale factor dynamically based on web dashboard calibration inputs
        scale_factor = (cal_distance / max(real_width, 1.0)) * 0.05
        real_distance_cm = pixel_distance * scale_factor
        
        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
        mid_point = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
        cv2.putText(frame, f"{real_distance_cm:.1f} cm", mid_point, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
    return frame