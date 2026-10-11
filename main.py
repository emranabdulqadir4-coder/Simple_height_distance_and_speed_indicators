from flask import Flask, render_template, Response, request, jsonify
import cv2
import height_engine
import distance_engine
import speed_engine

app = Flask(__name__)

# Keep track of which indicator menu option is selected
current_active_tool = 'height'

# Metric default state dictionaries
metrics = {
    "ref_height": 10.0,
    "real_width": 12.0,
    "cal_distance": 50.0,
    "fps": 30,
    "scale_factor": 0.15
}

@app.route('/')
def index():
    return render_template('indicators.html')

@app.route('/update-settings', methods=['POST'])
def update_settings():
    global current_active_tool
    data = request.json
    current_active_tool = data.get('tool', 'height')
    
    # Store settings modified by user web sliders
    for key in metrics:
        if key in data:
            metrics[key] = float(data[key])
            
    return jsonify({"status": "success", "active_tool": current_active_tool})

def generate_video_stream():
    # Connect directly to your local computer webcam pipeline (ID 0)
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Send the frame to the corresponding modular script based on active tab
            if current_active_tool == 'height':
                frame = height_engine.process_height_frame(frame, metrics['ref_height'])
            elif current_active_tool == 'distance':
                frame = distance_engine.process_distance_frame(frame, metrics['real_width'], metrics['cal_distance'])
            elif current_active_tool == 'speed':
                frame = speed_engine.process_speed_frame(frame, metrics['fps'], metrics['scale_factor'])
                
            # Compress and encode matrix array back into JPEG standard frame stream data
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Use multipart stream block transmission formatting
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Return a continuous live server stream to your HTML webpage image block
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
