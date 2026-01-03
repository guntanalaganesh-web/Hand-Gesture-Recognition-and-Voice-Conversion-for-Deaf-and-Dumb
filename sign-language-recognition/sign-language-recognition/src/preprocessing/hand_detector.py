"""
Hand Detection and Landmark Extraction using MediaPipe and OpenCV
Real-time hand tracking for sign language recognition
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import List, Tuple, Optional
import time


class HandDetector:
    """
    Hand detection and landmark extraction using MediaPipe
    Extracts 21 hand landmarks for gesture recognition
    """
    
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Landmark indices for specific fingers
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.FINGER_PIPS = [3, 6, 10, 14, 18]  # Proximal interphalangeal joints
        
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect hands in image and return annotated image with landmarks
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Tuple of (annotated_image, list of hand landmarks)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                   results.multi_handedness):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks, image.shape)
                hand_landmarks_list.append({
                    'landmarks': landmarks,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score
                })
        
        return annotated_image, hand_landmarks_list
    
    def _extract_landmarks(self, hand_landmarks, image_shape) -> np.ndarray:
        """Extract normalized landmark coordinates"""
        h, w, _ = image_shape
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            landmarks.extend([
                landmark.x,  # Already normalized (0-1)
                landmark.y,
                landmark.z
            ])
        
        return np.array(landmarks, dtype=np.float32)
    
    def get_hand_roi(self, image: np.ndarray, landmarks: np.ndarray, 
                     padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract hand region of interest from image
        
        Args:
            image: Input image
            landmarks: Hand landmarks array
            padding: Padding ratio around hand bbox
            
        Returns:
            Cropped hand ROI
        """
        h, w = image.shape[:2]
        
        # Reshape landmarks to (21, 3)
        points = landmarks.reshape(-1, 3)[:, :2]  # Get x, y only
        
        # Convert normalized coords to pixel coords
        points[:, 0] *= w
        points[:, 1] *= h
        
        # Get bounding box
        x_min, y_min = points.min(axis=0).astype(int)
        x_max, y_max = points.max(axis=0).astype(int)
        
        # Add padding
        pad_x = int((x_max - x_min) * padding)
        pad_y = int((y_max - y_min) * padding)
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)
        
        # Crop ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None
            
        return roi
    
    def count_fingers(self, landmarks: np.ndarray) -> int:
        """
        Count number of extended fingers
        
        Args:
            landmarks: Hand landmarks array (21*3)
            
        Returns:
            Number of extended fingers (0-5)
        """
        points = landmarks.reshape(21, 3)
        count = 0
        
        # Check thumb (special case - horizontal movement)
        if points[4][0] > points[3][0]:  # Right hand
            count += 1
        
        # Check other fingers (vertical movement)
        for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            if points[tip][1] < points[pip][1]:  # Y is inverted in image coords
                count += 1
        
        return count
    
    def get_finger_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate angles between finger joints for feature engineering"""
        points = landmarks.reshape(21, 3)
        angles = []
        
        # Define joint triplets for angle calculation
        joint_triplets = [
            (0, 1, 2), (1, 2, 3), (2, 3, 4),      # Thumb
            (0, 5, 6), (5, 6, 7), (6, 7, 8),      # Index
            (0, 9, 10), (9, 10, 11), (10, 11, 12), # Middle
            (0, 13, 14), (13, 14, 15), (14, 15, 16), # Ring
            (0, 17, 18), (17, 18, 19), (18, 19, 20)  # Pinky
        ]
        
        for p1, p2, p3 in joint_triplets:
            angle = self._calculate_angle(points[p1], points[p2], points[p3])
            angles.append(angle)
        
        return np.array(angles)
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate angle between three points"""
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()


class GestureSequenceCollector:
    """
    Collects sequences of hand landmarks for LSTM input
    Manages temporal data for dynamic gesture recognition
    """
    
    def __init__(self, sequence_length: int = 30, feature_dim: int = 63):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim  # 21 landmarks * 3 coordinates
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.is_recording = False
        self.gesture_data = []
        
    def add_frame(self, landmarks: Optional[np.ndarray]):
        """Add frame landmarks to sequence buffer"""
        if landmarks is not None:
            self.sequence_buffer.append(landmarks)
        else:
            # Add zero frame if no hand detected
            self.sequence_buffer.append(np.zeros(self.feature_dim))
    
    def get_sequence(self) -> Optional[np.ndarray]:
        """Get current sequence if complete"""
        if len(self.sequence_buffer) == self.sequence_length:
            return np.array(list(self.sequence_buffer))
        return None
    
    def start_recording(self):
        """Start recording a gesture"""
        self.is_recording = True
        self.gesture_data = []
        self.sequence_buffer.clear()
    
    def stop_recording(self) -> List[np.ndarray]:
        """Stop recording and return collected sequences"""
        self.is_recording = False
        data = self.gesture_data.copy()
        self.gesture_data = []
        return data
    
    def record_frame(self, landmarks: Optional[np.ndarray]):
        """Record frame during gesture capture"""
        if self.is_recording and landmarks is not None:
            self.gesture_data.append(landmarks)
    
    def clear(self):
        """Clear all buffers"""
        self.sequence_buffer.clear()
        self.gesture_data = []
        self.is_recording = False


class RealTimeProcessor:
    """
    Real-time video processing for sign language recognition
    Handles camera input, hand detection, and model inference
    """
    
    def __init__(self, model=None, camera_id: int = 0):
        self.detector = HandDetector()
        self.collector = GestureSequenceCollector()
        self.model = model
        self.camera_id = camera_id
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = None
        
    def start_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.start_time = time.time()
        
    def stop_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame
        
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        # Detect hands
        annotated_frame, hands = self.detector.detect_hands(frame)
        
        results = {
            'hands_detected': len(hands),
            'landmarks': None,
            'prediction': None,
            'confidence': 0.0
        }
        
        if hands:
            # Use first detected hand
            landmarks = hands[0]['landmarks']
            results['landmarks'] = landmarks
            
            # Add to sequence collector
            self.collector.add_frame(landmarks)
            
            # Get sequence for prediction
            sequence = self.collector.get_sequence()
            
            if sequence is not None and self.model is not None:
                # Predict gesture
                prediction = self.model.predict(sequence[np.newaxis, ...])
                results['prediction'] = np.argmax(prediction)
                results['confidence'] = float(np.max(prediction))
        else:
            self.collector.add_frame(None)
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        self.fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Add FPS to frame
        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, results
    
    def run(self, callback=None):
        """
        Main processing loop
        
        Args:
            callback: Optional function to call with results
        """
        self.start_camera()
        
        print("Press 'q' to quit, 'r' to start/stop recording")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            annotated_frame, results = self.process_frame(frame)
            
            # Display prediction if available
            if results['prediction'] is not None:
                text = f"Gesture: {results['prediction']} ({results['confidence']:.2f})"
                cv2.putText(annotated_frame, text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Call callback if provided
            if callback:
                callback(annotated_frame, results)
            
            # Display frame
            cv2.imshow('Sign Language Recognition', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.collector.is_recording:
                    data = self.collector.stop_recording()
                    print(f"Recorded {len(data)} frames")
                else:
                    self.collector.start_recording()
                    print("Recording started...")
        
        self.stop_camera()


if __name__ == "__main__":
    # Test hand detection
    print("Testing hand detection...")
    
    processor = RealTimeProcessor()
    processor.run()
