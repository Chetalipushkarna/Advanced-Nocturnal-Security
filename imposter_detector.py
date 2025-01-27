import numpy as np
import logging
from datetime import datetime, timedelta

class ImposterDetector:
    def __init__(self, thermal_analyzer, face_analyzer, movement_analyzer, config):
        self.thermal_analyzer = thermal_analyzer
        self.face_analyzer = face_analyzer
        self.movement_analyzer = movement_analyzer
        self.config = config
        
        # Cross-camera tracking
        self.camera_tracking = {}
        self.tracking_timeout = timedelta(minutes=5)  # Track objects for 5 minutes
    
    def detect_imposter(self, frame, detection_box, current_camera):
        try:
            # Analyze face concealment
            is_face_concealed, face_concealment_score = self.face_analyzer.analyze_face(frame)
            
            # Optional: Add other modality checks if needed
            is_imposter = is_face_concealed
            confidence = face_concealment_score
            
            # Cross-camera tracking
            if is_imposter:
                self._track_imposter_across_cameras(detection_box, current_camera)
            
            return is_imposter, confidence
        
        except Exception as e:
            logging.error(f"Imposter detection error: {str(e)}")
            return False, 0.0
    
    def _track_imposter_across_cameras(self, detection_box, current_camera):
        """Advanced cross-camera tracking of potential imposters"""
        # Get object center
        x1, y1, x2, y2 = detection_box[:4]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        current_time = datetime.now()
        
        # Clean up old tracking entries
        self._cleanup_tracking_entries(current_time)
        
        # Initialize camera tracking if not exists
        if current_camera not in self.camera_tracking:
            self.camera_tracking[current_camera] = {}
        
        # Add current detection
        object_id = len(self.camera_tracking[current_camera]) + 1
        self.camera_tracking[current_camera][object_id] = {
            'center': center,
            'timestamp': current_time
        }
        
        # Check for suspicious cross-camera movement
        for camera, objects in self.camera_tracking.items():
            if camera != current_camera:
                for obj_id, obj_data in objects.items():
                    distance = self._calculate_distance(center, obj_data['center'])
                    time_diff = (current_time - obj_data['timestamp']).total_seconds()
                    
                    # Suspicious if close distance and within tracking window
                    if distance < 100 and time_diff < 300:  # 100 pixel proximity, 5-minute window
                        logging.warning(f"Suspicious cross-camera movement detected. "
                f"Imposter detected in camera: {current_camera}")
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _cleanup_tracking_entries(self, current_time):
        """Remove old tracking entries"""
        for camera in list(self.camera_tracking.keys()):
            for obj_id in list(self.camera_tracking[camera].keys()):
                entry_time = self.camera_tracking[camera][obj_id]['timestamp']
                if (current_time - entry_time) > self.tracking_timeout:
                    del self.camera_tracking[camera][obj_id]


