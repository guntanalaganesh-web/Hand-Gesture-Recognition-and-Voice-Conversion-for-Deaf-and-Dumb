#!/usr/bin/env python3
"""
Hand Gesture Recognition and Voice Conversion System
Main entry point for the application

Usage:
    python main.py                    # Run real-time recognition
    python main.py --mode train       # Train models
    python main.py --mode demo        # Run demo with mock predictions
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_recognition(args):
    """Run real-time sign language recognition"""
    from src.inference.recognizer import SignLanguageRecognizer
    
    print("\n" + "="*60)
    print("  Hand Gesture Recognition and Voice Conversion System")
    print("  Real-time ASL Recognition with Speech Output")
    print("="*60 + "\n")
    
    recognizer = SignLanguageRecognizer(
        static_model_path=args.static_model,
        dynamic_model_path=args.dynamic_model,
        enable_speech=not args.no_speech,
        confidence_threshold=args.confidence
    )
    
    recognizer.run(camera_id=args.camera)


def run_training(args):
    """Train recognition models"""
    print("\n" + "="*60)
    print("  Training Sign Language Recognition Models")
    print("="*60 + "\n")
    
    if not args.data_dir:
        print("Error: --data-dir required for training")
        print("Usage: python main.py --mode train --data-dir /path/to/dataset")
        sys.exit(1)
    
    # Import training module
    from src.models.gesture_model import SignLanguageCNN, HybridCNNLSTM
    
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("\nTraining would start here with your dataset...")
    print("See src/training/train.py for full training pipeline")


def run_demo(args):
    """Run demo mode with mock predictions"""
    from src.inference.recognizer import SignLanguageRecognizer
    
    print("\n" + "="*60)
    print("  DEMO MODE - Hand Gesture Recognition")
    print("  (Using mock predictions without trained models)")
    print("="*60 + "\n")
    
    recognizer = SignLanguageRecognizer(
        static_model_path=None,  # No model - uses mock recognition
        dynamic_model_path=None,
        enable_speech=not args.no_speech,
        confidence_threshold=0.5
    )
    
    recognizer.run(camera_id=args.camera)


def main():
    parser = argparse.ArgumentParser(
        description='Hand Gesture Recognition and Voice Conversion System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run with camera (demo mode)
  python main.py --camera 1               # Use camera ID 1
  python main.py --no-speech              # Disable voice output
  python main.py --mode train --data-dir ./data  # Train models
        """
    )
    
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['recognize', 'train', 'demo'],
                       help='Operation mode')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--static-model', type=str,
                       help='Path to trained static gesture model')
    parser.add_argument('--dynamic-model', type=str,
                       help='Path to trained dynamic gesture model')
    parser.add_argument('--no-speech', action='store_true',
                       help='Disable text-to-speech output')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold for predictions')
    parser.add_argument('--data-dir', type=str,
                       help='Dataset directory (for training)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for models and logs')
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'recognize':
        run_recognition(args)
    else:  # demo
        run_demo(args)


if __name__ == "__main__":
    main()
