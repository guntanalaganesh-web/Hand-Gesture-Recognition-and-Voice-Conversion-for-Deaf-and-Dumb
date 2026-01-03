"""
Data Preprocessing and Feature Engineering for Sign Language Recognition
Includes data augmentation, normalization, and feature extraction
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import albumentations as A
from typing import Tuple, List, Optional
import os
import pickle
from tqdm import tqdm


class ImagePreprocessor:
    """
    Preprocessing pipeline for hand gesture images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64), grayscale: bool = True):
        self.target_size = target_size
        self.grayscale = grayscale
        self.scaler = StandardScaler()
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if self.grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        image = cv2.resize(image, self.target_size)
        
        # Apply histogram equalization for better contrast
        if len(image.shape) == 2:
            image = cv2.equalizeHist(image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension if grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        return image
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a batch of images"""
        return np.array([self.preprocess_image(img) for img in images])
    
    def apply_skin_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply skin color segmentation to isolate hand
        
        Args:
            image: BGR image
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(ycrcb, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def extract_hand_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract largest contour (hand) from binary mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        return largest_contour


class DataAugmentor:
    """
    Data augmentation for training robustness
    """
    
    def __init__(self, augment_probability: float = 0.5):
        self.augment_probability = augment_probability
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.3),
        ])
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single image"""
        if np.random.random() > self.augment_probability:
            return image
        
        # Ensure image is uint8 for albumentations
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        augmented = self.transform(image=image)
        result = augmented['image']
        
        # Convert back to float32 if needed
        if result.dtype == np.uint8:
            result = result.astype(np.float32) / 255.0
        
        return result
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Augment a sequence of frames with consistent transformations
        
        Args:
            sequence: Shape (seq_len, height, width, channels)
            
        Returns:
            Augmented sequence
        """
        # Apply same transformation to all frames
        replay_transform = A.ReplayCompose([
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        ])
        
        # Get transform params from first frame
        first_result = replay_transform(image=sequence[0])
        
        # Apply to all frames
        augmented_sequence = [first_result['image']]
        for frame in sequence[1:]:
            result = A.ReplayCompose.replay(first_result['replay'], image=frame)
            augmented_sequence.append(result['image'])
        
        return np.array(augmented_sequence)
    
    def augment_landmarks(self, landmarks: np.ndarray, 
                          noise_std: float = 0.01) -> np.ndarray:
        """
        Augment landmark coordinates
        
        Args:
            landmarks: Shape (21, 3) or (seq_len, 63)
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Augmented landmarks
        """
        # Add small random noise
        noise = np.random.normal(0, noise_std, landmarks.shape)
        augmented = landmarks + noise
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            augmented *= scale
        
        return augmented


class LandmarkPreprocessor:
    """
    Preprocessing for hand landmark data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks relative to wrist position
        Makes features translation-invariant
        
        Args:
            landmarks: Shape (21, 3) or flattened (63,)
            
        Returns:
            Normalized landmarks
        """
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(21, 3)
        
        # Center on wrist (landmark 0)
        wrist = landmarks[0].copy()
        centered = landmarks - wrist
        
        # Scale by palm size (distance from wrist to middle finger base)
        palm_size = np.linalg.norm(landmarks[9] - landmarks[0])
        if palm_size > 0:
            centered /= palm_size
        
        return centered.flatten()
    
    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Normalize a sequence of landmarks
        
        Args:
            sequence: Shape (seq_len, 63)
            
        Returns:
            Normalized sequence
        """
        return np.array([self.normalize_landmarks(frame) for frame in sequence])
    
    def fit_scaler(self, X: np.ndarray):
        """Fit StandardScaler on training data"""
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X)
        self.fitted = True
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization"""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit_scaler first.")
        
        original_shape = X.shape
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        
        X_transformed = self.scaler.transform(X)
        
        return X_transformed.reshape(original_shape)
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract additional features from landmarks
        
        Args:
            landmarks: Shape (21, 3)
            
        Returns:
            Feature vector including distances and angles
        """
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(21, 3)
        
        features = []
        
        # Fingertip distances from palm center
        palm_center = landmarks[[0, 5, 9, 13, 17]].mean(axis=0)
        fingertips = landmarks[[4, 8, 12, 16, 20]]
        
        for tip in fingertips:
            features.append(np.linalg.norm(tip - palm_center))
        
        # Distances between adjacent fingertips
        for i in range(len(fingertips) - 1):
            features.append(np.linalg.norm(fingertips[i] - fingertips[i+1]))
        
        # Thumb-finger distances
        thumb_tip = landmarks[4]
        for tip in fingertips[1:]:
            features.append(np.linalg.norm(thumb_tip - tip))
        
        # Finger curl (distance from tip to MCP)
        mcps = landmarks[[1, 5, 9, 13, 17]]
        tips = landmarks[[4, 8, 12, 16, 20]]
        for mcp, tip in zip(mcps, tips):
            features.append(np.linalg.norm(tip - mcp))
        
        return np.array(features)


class DatasetLoader:
    """
    Load and prepare sign language datasets
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.preprocessor = ImagePreprocessor()
        self.landmark_preprocessor = LandmarkPreprocessor()
        
    def load_image_dataset(self, split_ratio: float = 0.2) -> Tuple:
        """
        Load image dataset for static gesture recognition
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        images = []
        labels = []
        
        # Iterate through class folders
        for class_name in tqdm(os.listdir(self.data_dir), desc="Loading images"):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    processed = self.preprocessor.preprocess_image(img)
                    images.append(processed)
                    labels.append(class_name)
        
        X = np.array(images)
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=split_ratio * 2, stratify=y.argmax(axis=1), random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp.argmax(axis=1), random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_sequence_dataset(self, sequence_length: int = 30) -> Tuple:
        """
        Load sequence dataset for dynamic gesture recognition
        
        Returns:
            (X_train, X_val, y_train, y_val)
        """
        sequences = []
        labels = []
        
        # Load from numpy files
        for class_name in tqdm(os.listdir(self.data_dir), desc="Loading sequences"):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            for seq_file in os.listdir(class_path):
                if seq_file.endswith('.npy'):
                    seq_path = os.path.join(class_path, seq_file)
                    seq = np.load(seq_path)
                    
                    # Pad or truncate to fixed length
                    if len(seq) < sequence_length:
                        seq = np.pad(seq, ((0, sequence_length - len(seq)), (0, 0)))
                    else:
                        seq = seq[:sequence_length]
                    
                    # Normalize landmarks
                    seq = self.landmark_preprocessor.normalize_sequence(seq)
                    
                    sequences.append(seq)
                    labels.append(class_name)
        
        X = np.array(sequences)
        y = self.label_encoder.fit_transform(labels)
        y = to_categorical(y)
        
        # Fit and apply scaler
        self.landmark_preprocessor.fit_scaler(X)
        X = self.landmark_preprocessor.transform(X)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y.argmax(axis=1), random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    def save_preprocessors(self, filepath: str):
        """Save label encoder and scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'scaler': self.landmark_preprocessor.scaler
            }, f)
    
    def load_preprocessors(self, filepath: str):
        """Load label encoder and scaler"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.label_encoder = data['label_encoder']
            self.landmark_preprocessor.scaler = data['scaler']
            self.landmark_preprocessor.fitted = True


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing modules...")
    
    # Test image preprocessing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    preprocessor = ImagePreprocessor()
    processed = preprocessor.preprocess_image(test_image)
    print(f"Processed image shape: {processed.shape}")
    
    # Test augmentation
    augmentor = DataAugmentor()
    augmented = augmentor.augment_image(processed)
    print(f"Augmented image shape: {augmented.shape}")
    
    # Test landmark preprocessing
    test_landmarks = np.random.random((21, 3))
    landmark_proc = LandmarkPreprocessor()
    normalized = landmark_proc.normalize_landmarks(test_landmarks)
    print(f"Normalized landmarks shape: {normalized.shape}")
    
    features = landmark_proc.extract_features(test_landmarks)
    print(f"Extracted features: {len(features)}")
    
    print("\nPreprocessing tests completed!")
