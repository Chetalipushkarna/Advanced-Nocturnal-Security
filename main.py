# import cv2
# import time
# import logging
# from security_system import SecuritySystem
# import threading

# class CameraManager:
#     def __init__(self, cameras, reconnect_interval=30):
#         self.cameras = cameras
#         self.video_captures = {}
#         self.reconnect_interval = reconnect_interval
#         self.logger = logging.getLogger(__name__)
#         self.stop_event = threading.Event()
    
#     def _open_camera(self, cam_name, cam_id):
#         """Attempt to open camera with error handling"""
#         try:
#             cap = cv2.VideoCapture(cam_id)
#             if not cap.isOpened():
#                 self.logger.error(f"Could not open camera {cam_name}")
#                 return None
#             return cap
#         except Exception as e:
#             self.logger.error(f"Error opening camera {cam_name}: {str(e)}")
#             return None
    
#     def start_monitoring(self, security_system):
#         """Start monitoring with automatic reconnection"""
#         for cam_name, cam_id in self.cameras.items():
#             cap = self._open_camera(cam_name, cam_id)
#             if cap:
#                 self.video_captures[cam_name] = cap
        
#         monitoring_thread = threading.Thread(target=self._monitor_cameras, args=(security_system,))
#         monitoring_thread.start()
#         return monitoring_thread
    
#     def _monitor_cameras(self, security_system):
#         """Continuous camera monitoring with error recovery"""
#         while not self.stop_event.is_set():
#             try:
#                 # Capture frames from all cameras
#                 camera_frames = {}
#                 for cam_name, cap in list(self.video_captures.items()):
#                     ret, frame = cap.read()
#                     if not ret:
#                         self.logger.warning(f"Failed to capture frame from {cam_name}")
#                         # Attempt to reconnect
#                         cap.release()
#                         del self.video_captures[cam_name]
                        
#                         # Try to reopen camera
#                         new_cap = self._open_camera(cam_name, self.cameras[cam_name])
#                         if new_cap:
#                             self.video_captures[cam_name] = new_cap
#                         continue
                    
#                     camera_frames[cam_name] = frame
                
#                 # Process frames if any cameras are working
#                 if camera_frames:
#                     security_system.process_multi_camera_frames(camera_frames)
                
#                 # Optional: Display frames
#                 for cam_name, frame in camera_frames.items():
#                     cv2.imshow(cam_name, frame)
                
#                 # Break loop on 'q' key press
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
                
#             except Exception as e:
#                 self.logger.error(f"Error in camera monitoring: {str(e)}")
#                 time.sleep(self.reconnect_interval)
    
#     def stop(self):
#         """Stop monitoring and release resources"""
#         self.stop_event.set()
#         for cap in self.video_captures.values():
#             cap.release()
#         cv2.destroyAllWindows()

# def main():
#     # Setup logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     # Camera configuration
#     cameras = {
#         'cam1': "rtsp://admin:amazingct123@192.168.0.160:554/Streaming/Channels/101",
#         'cam2': "rtsp://admin:amazingct123@192.168.0.160:554/Streaming/Channels/201",
#         'cam3': 0  # Local webcam
#     }

#     # Initialize the security system
#     security_system = SecuritySystem()

#     # Create camera manager
#     camera_manager = CameraManager(cameras)

#     try:
#         # Start monitoring
#         monitoring_thread = camera_manager.start_monitoring(security_system)
        
#         # Wait for monitoring thread (or user to stop)
#         monitoring_thread.join()
    
#     except KeyboardInterrupt:
#         logger.info("Monitoring stopped by user")
    
#     finally:
#         # Ensure clean shutdown
#         camera_manager.stop()

# if __name__ == "__main__":
#     main()



import streamlit as st
import cv2
import time
import logging
from security_system import SecuritySystem
import threading

class CameraManager:
    def __init__(self, cameras, reconnect_interval=30):
        self.cameras = cameras
        self.video_captures = {}
        self.reconnect_interval = reconnect_interval
        self.logger = logging.getLogger(__name__)
        self.stop_event = threading.Event()
    
    def _open_camera(self, cam_name, cam_id):
        """Attempt to open camera with error handling"""
        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                self.logger.error(f"Could not open camera {cam_name}")
                return None
            return cap
        except Exception as e:
            self.logger.error(f"Error opening camera {cam_name}: {str(e)}")
            return None
    
    def start_monitoring(self, security_system):
        """Start monitoring with automatic reconnection"""
        for cam_name, cam_id in self.cameras.items():
            cap = self._open_camera(cam_name, cam_id)
            if cap:
                self.video_captures[cam_name] = cap
        
        monitoring_thread = threading.Thread(target=self._monitor_cameras, args=(security_system,))
        monitoring_thread.start()
        return monitoring_thread
    
    def _monitor_cameras(self, security_system):
        """Continuous camera monitoring with error recovery"""
        while not self.stop_event.is_set():
            try:
                # Capture frames from all cameras
                camera_frames = {}
                for cam_name, cap in list(self.video_captures.items()):
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.warning(f"Failed to capture frame from {cam_name}")
                        # Attempt to reconnect
                        cap.release()
                        del self.video_captures[cam_name]
                        
                        # Try to reopen camera
                        new_cap = self._open_camera(cam_name, self.cameras[cam_name])
                        if new_cap:
                            self.video_captures[cam_name] = new_cap
                        continue
                    
                    camera_frames[cam_name] = frame
                
                # Process frames if any cameras are working
                if camera_frames:
                    imposter_detected = security_system.process_multi_camera_frames(camera_frames)
                
                # Optional: Display frames
                for cam_name, frame in camera_frames.items():
                    if imposter_detected.get(cam_name, False):
                        # Draw bounding box on imposter
                        bbox = imposter_detected[cam_name]['bbox']
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    st.image(frame, channels="BGR")
                
                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                self.logger.error(f"Error in camera monitoring: {str(e)}")
                time.sleep(self.reconnect_interval)
    
    def stop(self):
        """Stop monitoring and release resources"""
        self.stop_event.set()
        for cap in self.video_captures.values():
            cap.release()
        cv2.destroyAllWindows()

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Camera configuration
    cameras = {
        'cam1': "rtsp://admin:cctv@321@192.168.1.64:554/Streaming/Channels/101",
        'cam2': "rtsp://admin:cctv@321@192.168.1.64:554/Streaming/Channels/101",
        'cam3': "rtsp://admin:cctv@321@192.168.1.64:554/Streaming/Channels/101" 
    }

    # Initialize the security system
    security_system = SecuritySystem()

    # Create camera manager
    camera_manager = CameraManager(cameras)

    try:
        # Start monitoring
        monitoring_thread = camera_manager.start_monitoring(security_system)
        
        # Wait for monitoring thread (or user to stop)
        monitoring_thread.join()
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    finally:
        # Ensure clean shutdown
        camera_manager.stop()

if __name__ == "__main__":
    main()