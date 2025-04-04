from flask import Flask, render_template, request, redirect, send_from_directory, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import cv2
import threading
import torch
from feature_extractor import FeatureExtractor, Config
from custom_utils import load_model
import numpy as np
import os
import time
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import threading
from face_detection import blur_frame
import json
from datetime import datetime






# Load the face detection Haar cascade
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.isfile(face_cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found at {face_cascade_path}")
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Add these global variables
processing_executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on CPU cores
anomaly_status = {}  # Track latest anomaly status for each camera
status_lock = threading.Lock()  # Ensure thread-safe updates

app = Flask(__name__)
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
FEEDBACK_FILE = "feedback.json"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "your_secret_key"  # Needed for session handling


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained anomaly detection model
MODEL_PATH = "model.pth"
anomaly_model = load_model(MODEL_PATH)

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

# Define camera sources (e.g., webcam indices or RTSP streams)
CAMERA_SOURCES = [0, 1]  # Replace with actual camera indices or URLs
FRAME_BATCH_SIZE = 8  # Number of frames to batch together for processing
# ANOMALY_THRESHOLD = 0.25
ANOMALY_THRESHOLD = 0.8

# Global variables for managing camera threads
camera_threads = {}
stop_event = threading.Event()

USER_FILE = "users.json"


if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({"admin": "pass123"}, f)


def load_feedback():
    """Load feedback data from JSON file, or return an empty list if the file doesn't exist."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            return json.load(file)
    return []

def save_feedback(new_entry):
    """Append new feedback to the file and save it."""
    feedback_data = load_feedback()
    feedback_data.append(new_entry)
    
    with open(FEEDBACK_FILE, "w") as file:
        json.dump(feedback_data, file, indent=4)

def process_frame_batch(frame_batch):
    """Process a batch of frames and detect anomalies."""
    features = feature_extractor.extract_features_from_frames(frame_batch)
    if features is None:
        return [0] * len(frame_batch)  # Return zero scores if extraction fails
    
    anomaly_scores = []
    with torch.no_grad():
        for feature in features:
            score = anomaly_model(feature.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')).item()
            anomaly_scores.append(score)
    return anomaly_scores

def capture_and_process(camera_id, source):
    """Capture frames, process resized frames, and display original frames with face blur."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    frame_buffer = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 / fps if fps > 0 else 0.033  # Approximate 30 FPS default

    # Start time of the stream
    stream_start_time = time.time()

    # Maintain a deque (sliding window) of the last 10 scores
    last_scores = deque(maxlen=10)

    # Batch counter
    batch_counter = 0

    def process_async(batch, camera_id, batch_num):
        """Process a batch of resized frames asynchronously."""
        # Record the time when the batch is sent for processing
        send_time = time.time() - stream_start_time
        print(f"Batch {batch_num} sent for processing at {send_time:.2f}s")

        # Resize frames for processing
        resized_batch = [cv2.resize(frame, (256, 256)) for frame in batch]
        
        # Extract features
        features = feature_extractor.extract_features_from_frames(resized_batch)
        if features is None:
            return

        # Detect anomalies
        with torch.no_grad():
            scores = anomaly_model(
                features.to('cuda' if torch.cuda.is_available() else 'cpu')
            ).cpu().numpy()

        # Record the time when the result is received
        receive_time = time.time() - stream_start_time
        print(f"Result for Batch {batch_num} received at {receive_time:.2f}s")

        # Print the anomaly scores with the timestamp
        print(f"Anomaly scores for Batch {batch_num}: {scores}")

        # Add the scores to the sliding window (last 10 scores)
        for score in scores:
            last_scores.append(score)

        # Check if 4 or more of the last 10 scores exceed the dynamic threshold
        anomaly_detected = sum(1 for score in last_scores if score > ANOMALY_THRESHOLD) >= 4

        # Print the final result (anomaly detected or not)
        print(f"Anomaly Detected for Batch {batch_num}: {anomaly_detected} (Threshold: {ANOMALY_THRESHOLD})")

        # Update anomaly status
        with status_lock:
            anomaly_status[camera_id] = anomaly_detected

    print("Stop event: ", stop_event.is_set())

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("breaking loop")
            break

        # Create a copy of the frame for display (blur faces)
        display_frame = frame.copy()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print("faces: ", faces)
        # Blur each detected face
        for (x, y, w, h) in faces:
            roi = display_frame[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            display_frame[y:y+h, x:x+w] = blurred_roi

        # Encode the blurred display frame
        _, buffer = cv2.imencode('.jpg', display_frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')

        # Buffer the original (unblurred) frame for processing
        frame_buffer.append(frame)

        # Submit batch for processing when ready
        if len(frame_buffer) == 32:
            # Copy the buffer to avoid thread conflicts
            batch_to_process = frame_buffer.copy()
            frame_buffer.clear()
            
            # Increment batch counter
            batch_counter += 1

            # Offload to thread pool
            processing_executor.submit(
                process_async, 
                batch_to_process, 
                camera_id,
                batch_counter  # Pass the batch number
            )

        # Get the latest anomaly status (default to False)
        with status_lock:
            current_status = anomaly_status.get(camera_id, False)

        # Send the blurred frame and anomaly status to the frontend
        socketio.emit('update_feed', {
            'camera_id': camera_id,
            'frame': encoded_frame,
            'anomaly_detected': int(current_status)  # Convert to int (1 or 0)
        })

        # Maintain approximate frame rate
        time.sleep(frame_interval)

    cap.release()

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): The name of the uploaded file.
    
    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}  # Add more extensions if needed
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def blur_faces_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print("Faces: ", faces)
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        frame[y:y+h, x:x+w] = blurred_roi
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    blurred_video_path = os.path.join("uploads", "blurred_" + os.path.basename(video_path))
    blurred_video_path_basename = "blurred_" + os.path.basename(video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # base_name = os.path.basename(video_path)
    # output_name = f"blurred_{os.path.splitext(base_name)[0]}.mp4"  # Force .mp4 extension
    # blurred_video_path = os.path.join("uploads", output_name)

    # output_filename = f"blurred_{os.path.basename(video_path).replace(' ', '_')}"
    # blurred_video_path = os.path.join("uploads", output_filename).replace("\\", "/")

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Browser-friendly H.264
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(blurred_video_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred_frame = blur_frame(frame, frame_width, frame_height)
        out.write(blurred_frame)
    
    cap.release()
    out.release()
    return blurred_video_path_basename





@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle login authentication."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        with open(USER_FILE, "r") as f:
            users = json.load(f)

        if users.get(username) == password:
            session["user"] = username
            return redirect(url_for("view_feedback"))

        return "Invalid credentials, try again!"

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Log out the user."""
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/view_feedback")
def view_feedback():
    """Display feedback only if logged in."""
    if "user" not in session:
        return redirect(url_for("login"))

    feedbacks = load_feedback()
    return render_template("view_feedback.html", feedbacks=feedbacks)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print("Filename: ", filename)
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename,
        mimetype='video/mp4'  # Explicit MIME type
    )


# @app.route("/try", methods=["GET", "POST"])
# def try_page():
#     if request.method == "POST":
#         if "file" not in request.files:
#             return redirect(request.url)
        
#         file = request.files["file"]
        
#         if file.filename == "":
#             return redirect(request.url)
        
#         if file and allowed_file(file.filename):
#             os.makedirs("uploads", exist_ok=True)
            
#             # video_path = os.path.join("uploads", secure_filename(file.filename))
#             video_path = os.path.join("uploads", secure_filename(file.filename)).replace("\\", "/")
#             file.save(video_path)
            
#             blurred_video_path = process_video(video_path)
#             # print("original path: ", blurred_video_path)
#             if not blurred_video_path:
#                 return render_template("index.html", error="Failed to process video.")
            
#             features = feature_extractor.extract_features(video_path)
#             if features is None:
#                 return render_template("index.html", error="Failed to extract features from video.")
            
#             total_anomalies = minor_anomalies = medium_anomalies = major_anomalies = 0
#             anomaly_logs = []

#             print(f"Original video path: {video_path}")
#             print(f"Blurred video path: {blurred_video_path}")
#             # print(f"Filename sent to template: {blurred_filename}")
            
#             for i in range(features.size(0)):
#                 with torch.no_grad():
#                     scores = anomaly_model(features[i].unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
#                     anomaly_score = scores.item()
                
#                 if anomaly_score > 0.7:
#                     anomaly_type = "Major Anomaly"
#                     major_anomalies += 1
#                 elif anomaly_score > 0.5:
#                     anomaly_type = "Medium Anomaly"
#                     medium_anomalies += 1
#                 elif anomaly_score > 0.25:
#                     anomaly_type = "Minor Anomaly"
#                     minor_anomalies += 1
#                 else:
#                     anomaly_type = "No Anomaly"
                
#                 if anomaly_type != "No Anomaly":
#                     total_anomalies += 1
                
#                 time_point = i * Config.stride_frames / 30
#                 anomaly_logs.append({
#                     "segment": i + 1,
#                     "time_point": f"{time_point:.2f} seconds",
#                     "anomaly_score": anomaly_score,
#                     "anomaly_type": anomaly_type
#                 })
            
#             report = {
#                 "total_anomalies": total_anomalies,
#                 "minor_anomalies": minor_anomalies,
#                 "medium_anomalies": medium_anomalies,
#                 "major_anomalies": major_anomalies,
#                 "anomaly_logs": anomaly_logs
#             }
#             print("Blurred video path: ", blurred_video_path)
#             return render_template("index.html", report=report, video_filename=blurred_video_path)
    
#     return render_template("index.html")


@app.route("/feedback", methods=["GET"])
def feedback():
    return render_template("feedback_page.html")


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """Handle feedback form submission and store it in JSON file."""
    data = request.form.to_dict()
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add unique timestamp
    
    save_feedback(data)
    
    return jsonify({"message": "Feedback submitted successfully!"}), 200


@app.route("/about-us", methods=["GET"])
def aboutUsPage():
    return render_template("about_us.html")

@app.route("/", methods=["GET"])
def landingPage():
    return render_template("landing_page.html")

@app.route("/try", methods=["GET", "POST"])
def tryScreen():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file selected")
            
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            blurred_filename = process_video(video_path)
            if not blurred_filename:
                return render_template("index.html", error="Error processing video")

            features = feature_extractor.extract_features(video_path)
            if features is None:
                return render_template("index.html", error="Error extracting features")

            total_anomalies = minor_anomalies = medium_anomalies = major_anomalies = 0
            anomaly_logs = []

            print(f"Original video path: {video_path}")
            # print(f"Blurred video path: {blurred_video_path}")
            # print(f"Filename sent to template: {blurred_filename}")
            
            for i in range(features.size(0)):
                with torch.no_grad():
                    scores = anomaly_model(features[i].unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
                    anomaly_score = scores.item()
                
                if anomaly_score > 0.7:
                    anomaly_type = "Major Anomaly"
                    major_anomalies += 1
                elif anomaly_score > 0.5:
                    anomaly_type = "Medium Anomaly"
                    medium_anomalies += 1
                elif anomaly_score > 0.25:
                    anomaly_type = "Minor Anomaly"
                    minor_anomalies += 1
                else:
                    anomaly_type = "No Anomaly"
                
                if anomaly_type != "No Anomaly":
                    total_anomalies += 1
                
                time_point = i * Config.stride_frames / 30
                anomaly_logs.append({
                    "segment": i + 1,
                    "time_point": f"{time_point:.2f} seconds",
                    "anomaly_score": anomaly_score,
                    "anomaly_type": anomaly_type
                })
            
            report = {
                "total_anomalies": total_anomalies,
                "minor_anomalies": minor_anomalies,
                "medium_anomalies": medium_anomalies,
                "major_anomalies": major_anomalies,
                "anomaly_logs": anomaly_logs
            }

            return render_template("index.html", 
                                video_filename=blurred_filename,
                                report=report)

        return render_template("index.html", error="Invalid file format")

    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True)