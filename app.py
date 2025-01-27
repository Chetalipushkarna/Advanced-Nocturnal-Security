import streamlit as st
import cv2
import numpy as np
from thermal_analyzer import ThermalAnalyzer
from movement_analyzer import MovementAnalyzer
from face_analyzer import FaceAnalyzer
from imposter_detector import ImposterDetector
from config import SecurityConfig
from ultralytics import YOLO
import time

# Initialize Security Config
config = SecurityConfig()

# Initialize Analyzers
thermal_analyzer = ThermalAnalyzer(config)
movement_analyzer = MovementAnalyzer(config)
face_analyzer = FaceAnalyzer(config)
imposter_detector = ImposterDetector(thermal_analyzer, face_analyzer, movement_analyzer, config)

# Initialize YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")  # Replace with your YOLO model path

# Streamlit Configuration
st.set_page_config(page_title="DefenSys", layout="wide", page_icon=":guardsman:")

# Sidebar Configuration
st.sidebar.header("Camera Settings")
camera_urls = [
    st.sidebar.text_input("Camera 1 URL", "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101"),
    st.sidebar.text_input("Camera 2 URL", "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101"),
    st.sidebar.text_input("Camera 3 URL", "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101"),
]
frame_width = st.sidebar.slider("Frame Width", 320, 1280, config.FRAME_WIDTH)
frame_height = st.sidebar.slider("Frame Height", 240, 720, config.FRAME_HEIGHT)
detection_interval = st.sidebar.slider("Detection Interval (s)", 0.1, 2.0, config.DETECTION_INTERVAL)

# Define Helper Functions
def get_detection_box(frame):
    """
    Uses YOLOv8 to detect a person and return the bounding box coordinates.
    """
    results = yolo_model(frame)
    detection_box = None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0:  # Class 0 represents person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detection_box = [int(x1), int(y1), int(x2), int(y2)]
                break
    return detection_box

def process_frame_with_detection(frame_rgb, detection_box, current_camera):
    """
    Processes the frame, runs imposter detection logic, and updates the frame with bounding boxes.
    """
    x1, y1, x2, y2 = detection_box

    # Run additional imposter checks
    is_imposter, total_score = imposter_detector.detect_imposter(
        frame_rgb, detection_box, current_camera=current_camera
    )

    if is_imposter:
        # Draw bounding box for imposter (red)
        frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"Imposter: {total_score:.2f}"
        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        # Draw bounding box for civilian (green)
        frame_rgb = cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "Civilian"
        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame_rgb, is_imposter

# Main App
st.title("DefenSys")
run_cameras = st.button("Start All Cameras")

if run_cameras:
    placeholders = [st.empty() for _ in camera_urls]  # Placeholders for the video feeds

    while True:
        for idx, camera_url in enumerate(camera_urls):
            if not camera_url.strip():
                continue  # Skip empty camera URLs

            cap = cv2.VideoCapture(camera_url)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

            if not cap.isOpened():
                st.error(f"Failed to connect to Camera {idx + 1}. Check the RTSP URL.")
                continue

            ret, frame = cap.read()
            if not ret:
                st.error(f"Failed to capture video frame from Camera {idx + 1}.")
                cap.release()
                continue

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_box = get_detection_box(frame_rgb)

            if detection_box:
                # Process frame and run detection
                frame_rgb, is_imposter = process_frame_with_detection(
                    frame_rgb, detection_box, current_camera=idx + 1
                )

                # Display only if an imposter is detected
                if is_imposter:
                    placeholders[idx].image(
                        frame_rgb,
                        caption=f"Imposter Detected in Camera {idx + 1}",
                        channels="RGB",
                        use_container_width=True,
                    )
                    st.write(f"**Camera {idx + 1}: Imposter Detected!**")
                else:
                    placeholders[idx].image(
                        frame_rgb,
                        caption=f"Civilian Detected in Camera {idx + 1}",
                        channels="RGB",
                        use_container_width=True,
                    )

            else:
                placeholders[idx].empty()  # Clear the placeholder if no detection

            time.sleep(detection_interval)  # Delay to control processing rate
            cap.release()
