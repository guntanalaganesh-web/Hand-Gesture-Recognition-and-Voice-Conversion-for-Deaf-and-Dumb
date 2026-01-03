"""
Real-time Sign Language Recognition and Speech Conversion Pipeline
Main inference module integrating all components
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
from typing import Optional, Dict, List, Tuple
import tensorflow as tf

from preprocessing.hand_detector import HandDetector, GestureSequenceCollector
from preprocessing.data_pipeline import LandmarkPreprocessor, ImagePreprocessor
from inference.text_to_speech import SpeechSynthesizer, GestureToSpeech
from models.gesture_model import HybridCNNLSTM, SignLanguageCNN

import sys
sys.path.append('..')


class SignLanguageRecognizer:
    """
    Complete Sign Language Recognition System
    Combines hand detection, gesture recognition, and speech synthesis
    """
    
    # ASL Alphabet mapping
    ASL_ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # Common ASL words/phrases
    ASL_WORDS = [
        'hello', 'goodbye', 'please', 'thanks', 'sorry',
        'yes', 'no', 'help', 'love', 'understand',
        'name', 'what', 'where', 'when', 'why', 'how',
        'good', 'bad', 'happy', 'sad', 'hungry', 'thirsty',
        'hot', 'cold', 'big', 'small', 'more', 'done'
    ]
    
    def __init__(self, 
                 static_model_path: Optional[str] = None,
                 dynamic_model_path: Optional[str] = None,
                 enable_speech: bool = True,
                 confidence_threshold: float = 0.7):
        """
        Initialize the recognition system
        
        Args:
            static_model_path: Path to CNN model for static gestures
            dynamic_model_path: Path to LSTM model for dynamic gestures
            enable_speech: Enable text-to-speech output
            confidence_threshold: Minimum confidence for predictions
        """
        print("Initializing Sign Language Recognition System...")
        
        # Initialize components
        self.hand_detector = HandDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.sequence_collector = GestureSequenceCollector(
            sequence_length=30,
            feature_dim=63
        )
        
        self.landmark_preprocessor = LandmarkPreprocessor()
        self.image_preprocessor = ImagePreprocessor(target_size=(64, 64))
        
        # Load models
        self.static_model = self._load_model(static_model_path, 'static')
        self.dynamic_model = self._load_model(dynamic_model_path, 'dynamic')
        
        # Speech synthesis
        self.enable_speech = enable_speech
        if enable_speech:
            self.synthesizer = SpeechSynthesizer()
            self.gesture_to_speech = GestureToSpeech(self.synthesizer)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        
        # State tracking
        self.current_mode = 'static'  # 'static' or 'dynamic'
        self.prediction_history = deque(maxlen=10)
        self.stable_prediction = None
        self.prediction_count = 0
        self.stability_threshold = 3  # Number of consistent predictions needed
        
        # Performance metrics
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        self.inference_times = deque(maxlen=30)
        
        print("System initialized successfully!")
    
    def _load_model(self, model_path: Optional[str], model_type: str):
        """Load a trained model"""
        if model_path is None:
            print(f"No {model_type} model path provided. Using placeholder.")
            return None
        
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded {model_type} model from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for gesture recognition
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Dictionary with results and annotated frame
        """
        start_time = time.time()
        
        # Detect hands
        annotated_frame, hands = self.hand_detector.detect_hands(frame)
        
        results = {
            'hands_detected': len(hands),
            'gesture': None,
            'confidence': 0.0,
            'mode': self.current_mode,
            'landmarks': None
        }
        
        if not hands:
            self.sequence_collector.add_frame(None)
            return results, annotated_frame
        
        # Get landmarks from first detected hand
        hand_data = hands[0]
        landmarks = hand_data['landmarks']
        results['landmarks'] = landmarks
        
        # Add to sequence collector
        self.sequence_collector.add_frame(landmarks)
        
        # Perform recognition based on mode
        if self.current_mode == 'static':
            gesture, confidence = self._recognize_static(frame, landmarks)
        else:
            gesture, confidence = self._recognize_dynamic()
        
        results['gesture'] = gesture
        results['confidence'] = confidence
        
        # Update prediction stability
        self._update_stability(gesture, confidence)
        
        # Convert to speech if stable prediction
        if (self.enable_speech and 
            self.stable_prediction is not None and 
            confidence >= self.confidence_threshold):
            self.gesture_to_speech.process_gesture(
                self.stable_prediction, 
                confidence
            )
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Draw info on frame
        annotated_frame = self._draw_info(annotated_frame, results)
        
        return results, annotated_frame
    
    def _recognize_static(self, frame: np.ndarray, 
                          landmarks: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize static gesture using CNN"""
        if self.static_model is None:
            return self._mock_recognition(landmarks)
        
        # Extract hand ROI
        roi = self.hand_detector.get_hand_roi(frame, landmarks)
        if roi is None:
            return None, 0.0
        
        # Preprocess
        processed = self.image_preprocessor.preprocess_image(roi)
        processed = np.expand_dims(processed, axis=0)
        
        # Predict
        predictions = self.static_model.predict(processed, verbose=0)
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx])
        
        # Map to label
        if class_idx < len(self.ASL_ALPHABET):
            gesture = self.ASL_ALPHABET[class_idx]
        else:
            gesture = f"gesture_{class_idx}"
        
        return gesture, confidence
    
    def _recognize_dynamic(self) -> Tuple[Optional[str], float]:
        """Recognize dynamic gesture using LSTM"""
        sequence = self.sequence_collector.get_sequence()
        
        if sequence is None or self.dynamic_model is None:
            return None, 0.0
        
        # Normalize sequence
        normalized = self.landmark_preprocessor.normalize_sequence(sequence)
        normalized = np.expand_dims(normalized, axis=0)
        
        # Predict
        predictions = self.dynamic_model.predict(normalized, verbose=0)
        class_idx = np.argmax(predictions)
        confidence = float(predictions[0][class_idx])
        
        # Map to label
        if class_idx < len(self.ASL_WORDS):
            gesture = self.ASL_WORDS[class_idx]
        else:
            gesture = f"word_{class_idx}"
        
        return gesture, confidence
    
    def _mock_recognition(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """Mock recognition for testing without trained models"""
        # Count fingers as simple gesture
        finger_count = self.hand_detector.count_fingers(landmarks)
        
        mock_gestures = {
            0: ('fist', 0.9),
            1: ('point', 0.85),
            2: ('peace', 0.88),
            3: ('three', 0.82),
            4: ('four', 0.80),
            5: ('open_hand', 0.92)
        }
        
        return mock_gestures.get(finger_count, ('unknown', 0.5))
    
    def _update_stability(self, gesture: Optional[str], confidence: float):
        """Track prediction stability for smoother output"""
        if gesture is None or confidence < self.confidence_threshold:
            self.prediction_count = 0
            return
        
        self.prediction_history.append(gesture)
        
        # Check if prediction is stable
        if len(self.prediction_history) >= self.stability_threshold:
            recent = list(self.prediction_history)[-self.stability_threshold:]
            if all(p == gesture for p in recent):
                if self.stable_prediction != gesture:
                    self.stable_prediction = gesture
                    self.prediction_count = 1
                else:
                    self.prediction_count += 1
    
    def _draw_info(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw recognition info on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # FPS
        if self.frame_times:
            fps = len(self.frame_times) / sum(self.frame_times) if sum(self.frame_times) > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Mode
        mode_color = (0, 255, 255) if self.current_mode == 'static' else (255, 255, 0)
        cv2.putText(frame, f"Mode: {self.current_mode.upper()}", (200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Gesture prediction
        if results['gesture']:
            gesture_text = f"Gesture: {results['gesture']}"
            confidence_text = f"Confidence: {results['confidence']:.2%}"
            
            color = (0, 255, 0) if results['confidence'] >= self.confidence_threshold else (0, 165, 255)
            
            cv2.putText(frame, gesture_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, confidence_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Stable prediction indicator
        if self.stable_prediction:
            cv2.putText(frame, f"[{self.stable_prediction}]", (w - 200, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Hands detected
        hands_text = f"Hands: {results['hands_detected']}"
        cv2.putText(frame, hands_text, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def toggle_mode(self):
        """Toggle between static and dynamic recognition modes"""
        if self.current_mode == 'static':
            self.current_mode = 'dynamic'
            print("Switched to DYNAMIC mode (word recognition)")
        else:
            self.current_mode = 'static'
            print("Switched to STATIC mode (letter recognition)")
        
        # Clear state
        self.sequence_collector.clear()
        self.prediction_history.clear()
        self.stable_prediction = None
    
    def run(self, camera_id: int = 0):
        """
        Main recognition loop
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n=== Sign Language Recognition System ===")
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Toggle mode (static/dynamic)")
        print("  's' - Speak last prediction")
        print("  'c' - Clear prediction history")
        print("=========================================\n")
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                results, annotated_frame = self.process_frame(frame)
                
                # Track frame time
                self.frame_times.append(time.time() - frame_start)
                
                # Display
                cv2.imshow('Sign Language Recognition', annotated_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.toggle_mode()
                elif key == ord('s') and self.enable_speech and self.stable_prediction:
                    self.synthesizer.speak(self.stable_prediction, interrupt=True)
                elif key == ord('c'):
                    self.prediction_history.clear()
                    self.stable_prediction = None
                    if self.enable_speech:
                        self.gesture_to_speech.clear_history()
                    print("Cleared prediction history")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            if self.enable_speech:
                self.synthesizer.stop()
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        
        return {
            'avg_inference_time_ms': avg_inference * 1000,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'fps': 1 / avg_frame_time if avg_frame_time > 0 else 0,
            'total_predictions': len(self.prediction_history),
            'stable_prediction': self.stable_prediction
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sign Language Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--static-model', type=str, help='Path to static gesture model')
    parser.add_argument('--dynamic-model', type=str, help='Path to dynamic gesture model')
    parser.add_argument('--no-speech', action='store_true', help='Disable speech output')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    recognizer = SignLanguageRecognizer(
        static_model_path=args.static_model,
        dynamic_model_path=args.dynamic_model,
        enable_speech=not args.no_speech,
        confidence_threshold=args.confidence
    )
    
    # Run recognition
    recognizer.run(camera_id=args.camera)
    
    # Print final statistics
    stats = recognizer.get_statistics()
    print("\n=== Final Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
