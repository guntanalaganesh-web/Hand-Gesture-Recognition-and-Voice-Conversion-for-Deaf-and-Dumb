"""
Flask Web Application for Sign Language Recognition
Provides web interface for gesture recognition
"""

from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference.recognizer import SignLanguageRecognizer

app = Flask(__name__)
CORS(app)

# Global recognizer instance
recognizer = None

def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = SignLanguageRecognizer(
            enable_speech=False,  # Disable speech for web
            confidence_threshold=0.6
        )
    return recognizer


def generate_frames():
    """Video streaming generator"""
    rec = get_recognizer()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        results, annotated_frame = rec.process_frame(frame)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def status():
    rec = get_recognizer()
    return jsonify({
        'status': 'running',
        'mode': rec.current_mode,
        'prediction': rec.stable_prediction,
        'stats': rec.get_statistics()
    })


@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    rec = get_recognizer()
    rec.toggle_mode()
    return jsonify({'mode': rec.current_mode})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
