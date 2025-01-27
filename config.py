import os

class SecurityConfig:
    """Enhanced configuration for imposter detection"""
    def __init__(self):
        # Camera settings
        self.CAMERA_IDS = {
            'cam1': "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101",
            'cam2': "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101",
            'cam3': "rtsp://admin:cctv@321@192.168.1.65:554/Streaming/Channels/101"
        }

        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.DETECTION_INTERVAL = 0.1
        
        # Detection thresholds
        self.PERSON_DETECTION_THRESHOLD = 0.85
        self.FACE_DETECTION_THRESHOLD = 0.75
        self.THERMAL_ANOMALY_THRESHOLD = 0.8
        self.MOVEMENT_ANOMALY_THRESHOLD = 0.7
        self.FACE_CONCEALMENT_THRESHOLD = 0.75
        
        # Paths
        self.PERSON_MODEL_PATH = 'model/mobilenet_v2_thermal.tflite'
        self.FACE_MODEL_PATH = 'model/face_detection_model.tflite'
        self.SAVE_PATH = 'detections/'

        # Thermal analysis
        self.NORMAL_TEMP_RANGE = (35.0, 37.5)  # Normal human temperature range
        self.THERMAL_WINDOW_SIZE = 30  # Frames to analyze for thermal patterns
        
        # Movement analysis
        self.MOVEMENT_HISTORY = 50  # Frames to track for movement analysis
        self.NORMAL_SPEED_RANGE = (0.1, 2.0)  # meters per second
        
        # Create save directory
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
