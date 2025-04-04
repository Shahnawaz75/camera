import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from tqdm import tqdm

def get_video_path():
    """Get video path with fallback to CLI input"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        return path
    except Exception as e:
        print(f"GUI selection failed: {e}. Using CLI input.")
        return input("Enter video file path: ").strip('"')

def validate_environment():
    """Check required dependencies"""
    try:
        cv2.__version__
        mp.__version__
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install requirements with:")
        print("pip install opencv-python mediapipe tqdm numpy")
        return False

def main():
    if not validate_environment():
        return

    input_path = get_video_path()
    if not input_path or not os.path.exists(input_path):
        print("Invalid file path")
        return

    print(f"\nProcessing: {os.path.basename(input_path)}")
    output_path = os.path.splitext(input_path)[0] + "_blurred.mp4"

    try:
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3
        )
        print("Face detection model loaded successfully")

        # Video setup
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Failed to open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            raise IOError("Failed to create output video")

        print(f"Original resolution: {w}x{h}")
        print(f"Estimated processing time: {total_frames/fps/60:.1f} minutes")

        # Processing loop
        progress = tqdm(total=total_frames, desc="Processing", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x = int(box.xmin * w)
                    y = int(box.ymin * h)
                    width = int(box.width * w)
                    height = int(box.height * h)

                    # Expand and validate region
                    x = max(0, x - 20)
                    y = max(0, y - 20)
                    width = min(w - x, width + 40)
                    height = min(h - y, height + 40)

                    # Apply blur
                    face_roi = frame[y:y+height, x:x+width]
                    if face_roi.size > 0:
                        blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
                        frame[y:y+height, x:x+width] = blurred

            # Write frame
            out.write(frame)
            progress.update(1)

        print("\nProcessing completed successfully")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        cap.release() if 'cap' in locals() else None
        out.release() if 'out' in locals() else None
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    input("Press Enter to exit...")