import cv2
import logging
import threading
import numpy as np
from config import SecurityConfig
from thermal_analyzer import ThermalAnalyzer
from face_analyzer import FaceAnalyzer
from movement_analyzer import MovementAnalyzer
from imposter_detector import ImposterDetector
from ultralytics import YOLO
import time

class SecuritySystem:
    def __init__(self):
        # Initialize configuration
        self.config = SecurityConfig()
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize analyzers
        self.thermal_analyzer = ThermalAnalyzer(self.config)
        self.face_analyzer = FaceAnalyzer(self.config)
        self.movement_analyzer = MovementAnalyzer(self.config)
        
        # Initialize imposter detector
        self.imposter_detector = ImposterDetector(
            self.thermal_analyzer, 
            self.face_analyzer, 
            self.movement_analyzer,
            self.config
        )
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def process_multi_camera_frames(self, camera_frames):
        """
        Process frames from multiple cameras
        
        Args:
            camera_frames (dict): Dictionary of camera names and their frames
        """
        for camera_name, frame in camera_frames.items():
            try:
                # Detect persons
                persons = self._detect_persons(frame)
                
                for person in persons:
                    # Analyze each detected person
                    is_imposter, confidence = self._analyze_person(frame, person, camera_name)
                    
                    # Visualize results
                    self._visualize_detection(frame, person, is_imposter, confidence)
                
            except Exception as e:
                self.logger.error(f"Error processing camera {camera_name}: {str(e)}")

    def _detect_persons(self, frame):
        """
        Detect persons in a frame using YOLOv8
        
        Args:
            frame (numpy.ndarray): Video frame
        
        Returns:
            list: List of bounding boxes for detected persons
        """
        # Detect objects in the frame
        results = self.model(frame)

        # Extract bounding boxes for persons (class index 0)
        person_bboxes = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Class index 0 corresponds to 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
                    person_bboxes.append({'bbox': [x1, y1, x2, y2]})
        
        return person_bboxes

    def _analyze_person(self, frame, person, camera_name):
        """
        Analyze a detected person
        
        Args:
            frame (numpy.ndarray): Video frame
            person (dict): Person detection information
            camera_name (str): Name of the camera
        
        Returns:
            tuple: (is_imposter, confidence)
        """
        # Detect imposter using multi-modal analysis
        is_imposter, confidence = self.imposter_detector.detect_imposter(
            frame, 
            person['bbox'], 
            camera_name
        )
        
        return is_imposter, confidence

    def _visualize_detection(self, frame, person, is_imposter, confidence):
        """
        Visualize detection results on the frame
        
        Args:
            frame (numpy.ndarray): Video frame
            person (dict): Person detection information
            is_imposter (bool): Whether the person is an imposter
            confidence (float): Imposter detection confidence
        """
        bbox = person['bbox']
        
        # Choose color based on imposter status
        color = (0, 0, 255) if is_imposter else (0, 255, 0)  # Red for imposter, Green for normal
        
        # Draw bounding box
        cv2.rectangle(
            frame, 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            color, 
            2
        )
        
        # Add label
        label = f"Imposter: {confidence:.2f}" if is_imposter else "Normal"
        cv2.putText(
            frame, 
            label, 
            (int(bbox[0]), int(bbox[1]-10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            color, 
            2
        )