import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

# Global parameter cache to remember the last position across web frame intervals
prev_x = None

def process_speed_frame(frame, fps, scale_factor):
    global prev_x
    h, w, _ = frame.shape
    
    results = model(frame, verbose=False)
    
    for r in results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            joints = r.keypoints.xy[0].cpu().numpy()
            
            # YOLO Keypoint ID: 11 = Left Hip (Central body mass anchor point)
            if len(joints) > 11:
                current_x = joints[11][0]
                
                if current_x > 0:
                    if prev_x is not None:
                        # Find the structural pixel shift value
                        pixel_displacement = abs(current_x - prev_x)
                        
                        # Velocity Math Formula: Speed = (Pixels * Scale Factor) * Framerate
                        speed_cms = (pixel_displacement * scale_factor) * fps
                        speed_kmh = (speed_cms / 100) * 3.6
                        
                        cv2.putText(frame, f"YOLO Identified Velocity: {speed_kmh:.2f} km/h", (30, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Analyzing baseline... Stay in frame", (30, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    
                    prev_x = current_x
                    
    return frame
