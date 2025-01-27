import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MovementAnalyzer:
    """Advanced movement analyzer with path tracking and anomaly detection"""
    def __init__(self, config):
        self.config = config
        
        # Path tracking parameters
        self.path_history = {}  # Store paths for each detected object
        self.max_path_length = 100  # Maximum number of points to store in a path
        
        # Movement analysis parameters
        self.movement_history = deque(maxlen=config.MOVEMENT_HISTORY)
        self.last_position = None
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    
    def analyze_movement(self, frame, detection_box):
        """
        Analyze movement patterns and track object path
        
        Args:
            frame (numpy.ndarray): Current video frame
            detection_box (list): Bounding box of detected object [x1, y1, x2, y2]
        
        Returns:
            tuple: (is_movement_anomalous, movement_score, path_info)
        """
        try:
            if detection_box is None:
                return False, 0.0, None
            
            # Calculate current position
            current_position = self._calculate_position(detection_box)
            
            # Track path
            path_info = self._track_path(current_position)
            
            # Extract movement features
            movement_features = self._extract_movement_features(current_position)
            
            if movement_features is None:
                return False, 0.0, path_info
            
            # Update movement history
            self.movement_history.append(movement_features)
            
            # Require minimum history for analysis
            if len(self.movement_history) < 10:
                return False, 0.0, path_info
            
            # Detect movement anomalies
            movement_data = np.array(list(self.movement_history))
            anomaly_scores = self.isolation_forest.fit_predict(movement_data)
            
            # Calculate anomaly probability
            anomaly_prob = np.mean(anomaly_scores == -1)
            is_anomalous = anomaly_prob > self.config.MOVEMENT_ANOMALY_THRESHOLD
            
            # Update last position
            self.last_position = current_position
            
            return is_anomalous, anomaly_prob, path_info
        
        except Exception as e:
            logger.error(f"Error in movement analysis: {str(e)}")
            return False, 0.0, None
    
    def _calculate_position(self, detection_box):
        """Calculate center position of detection"""
        x1, y1, x2, y2 = detection_box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _track_path(self, current_position):
        """
        Track object movement path
        
        Args:
            current_position (tuple): Current (x, y) position
        
        Returns:
            dict: Path tracking information
        """
        # Generate a unique object ID (in a real scenario, use object tracking)
        object_id = 0  # Simplified for this example
        
        # Initialize path if not exists
        if object_id not in self.path_history:
            self.path_history[object_id] = deque(maxlen=self.max_path_length)
        
        # Add current position to path
        self.path_history[object_id].append(current_position)
        
        # Analyze path characteristics
        path = list(self.path_history[object_id])
        
        # Calculate path metrics
        path_info = {
            'total_distance': self._calculate_total_path_distance(path),
            'path_length': len(path),
            'start_point': path[0] if path else None,
            'end_point': path[-1] if path else None,
            'is_linear': self._check_path_linearity(path),
            'direction_changes': self._count_direction_changes(path)
        }
        
        return path_info
    
    def _calculate_total_path_distance(self, path):
        """Calculate total distance traveled"""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance
    
    def _check_path_linearity(self, path, threshold=0.2):
        """
        Check if path is relatively linear
        
        Args:
            path (list): List of (x, y) coordinates
            threshold (float): Tolerance for deviation from linear path
        
        Returns:
            bool: True if path is linear, False otherwise
        """
        if len(path) < 3:
            return True
        
        # Calculate line of best fit
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        # Use numpy's polyfit to get line of best fit
        try:
            coefficient = np.polyfit(x, y, 1)
            predicted_y = np.poly1d(coefficient)(x)
            
            # Calculate average deviation
            deviation = np.mean(np.abs(np.array(y) - predicted_y))
            
            return deviation < threshold
        except Exception:
            return False
    
    def _count_direction_changes(self, path):
        """
        Count number of significant direction changes in path
        
        Args:
            path (list): List of (x, y) coordinates
        
        Returns:
            int: Number of direction changes
        """
        if len(path) < 3:
            return 0
        
        direction_changes = 0
        for i in range(2, len(path)):
            # Calculate vectors
            vector1 = (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1])
            vector2 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            
            # Calculate angle between vectors
            dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
            magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
            magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                continue
            
            cos_angle = dot_product / (magnitude1 * magnitude2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # Consider significant direction change
            if angle > np.pi/4:  # More than 45-degree change
                direction_changes += 1
        
        return direction_changes
    
    def _extract_movement_features(self, current_position):
        """Extract features based on movement"""
        if self.last_position is None:
            return None
        
        # Calculate movement parameters
        delta_x = current_position[0] - self.last_position[0]
        delta_y = current_position[1] - self.last_position[1]
        
        # Speed calculation
        speed = np.sqrt(delta_x**2 + delta_y**2)
        
        # Angle of movement
        movement_angle = np.arctan2(delta_y, delta_x)
        
        return np.array([speed, movement_angle])