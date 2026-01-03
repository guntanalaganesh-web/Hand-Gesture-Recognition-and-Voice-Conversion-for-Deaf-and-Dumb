# 🤟 Hand Gesture Recognition and Voice Conversion System

**Real-time Sign Language Recognition using Deep Learning**

A pioneering system that translates ASL (American Sign Language) gestures to speech in real-time, achieving **85% accuracy** using CNN-LSTM hybrid architectures.

> 📄 **Published Research**: Journal of Positive School Psychology

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Sign Language Recognition Pipeline                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │  Camera  │───▶│ Hand Detection│───▶│  Feature    │───▶│   CNN     │  │
│  │  Input   │    │  (MediaPipe)  │    │  Extraction │    │  (Static) │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └─────┬─────┘  │
│                          │                                     │        │
│                          │           ┌─────────────┐          │        │
│                          └──────────▶│   LSTM      │◀─────────┘        │
│                                      │  (Dynamic)  │                   │
│                                      └──────┬──────┘                   │
│                                             │                          │
│                    ┌────────────────────────┼────────────────────────┐ │
│                    │                        ▼                        │ │
│                    │              ┌─────────────────┐                │ │
│                    │              │  Gesture Class  │                │ │
│                    │              │   Prediction    │                │ │
│                    │              └────────┬────────┘                │ │
│                    │                       │                         │ │
│                    │                       ▼                         │ │
│                    │              ┌─────────────────┐                │ │
│                    │              │  Text-to-Speech │                │ │
│                    │              │    (pyttsx3)    │                │ │
│                    │              └────────┬────────┘                │ │
│                    │                       │                         │ │
│                    │                       ▼                         │ │
│                    │              ┌─────────────────┐                │ │
│                    │              │  Audio Output   │                │ │
│                    │              │    (Speaker)    │                │ │
│                    │              └─────────────────┘                │ │
│                    └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Tech Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow, PyTorch, Keras |
| **Computer Vision** | OpenCV, MediaPipe |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Speech Synthesis** | pyttsx3, gTTS, Edge-TTS |
| **Visualization** | Matplotlib, Seaborn |

## 📊 Model Performance

| Model | Accuracy | Use Case |
|-------|----------|----------|
| **CNN** | 87% | Static gestures (alphabet) |
| **LSTM** | 82% | Dynamic gestures (words) |
| **Hybrid CNN-LSTM** | **85%** | Combined recognition |
| **Attention LSTM** | 84% | Long sequences |

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- Webcam
- ~4GB RAM

### Installation

```bash
# Clone/extract project
cd sign-language-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (No Training Required)

```bash
# Run with webcam - uses mock predictions
python main.py

# Or explicitly:
python main.py --mode demo
```

### Controls
- **'q'** - Quit
- **'m'** - Toggle static/dynamic mode
- **'s'** - Speak last prediction
- **'c'** - Clear history

## 📁 Project Structure

```
sign-language-recognition/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── src/
│   ├── models/
│   │   └── gesture_model.py   # CNN, LSTM, Hybrid models
│   ├── preprocessing/
│   │   ├── hand_detector.py   # MediaPipe hand detection
│   │   └── data_pipeline.py   # Data preprocessing
│   ├── inference/
│   │   ├── recognizer.py      # Main recognition pipeline
│   │   └── text_to_speech.py  # TTS conversion
│   └── utils/
├── data/                      # Dataset storage
├── notebooks/                 # Jupyter notebooks
└── webapp/                    # Flask web interface
```

## 🎯 Features

### Hand Detection (MediaPipe)
- 21 hand landmarks tracking
- Real-time 30+ FPS
- Multi-hand support
- Finger counting

### Gesture Recognition
- **Static Mode**: Alphabet recognition (A-Z)
- **Dynamic Mode**: Word/phrase recognition
- Confidence scoring
- Prediction stability filtering

### Speech Output
- Multiple TTS engines:
  - **pyttsx3** - Offline (fastest)
  - **gTTS** - Google TTS (high quality)
  - **Edge-TTS** - Microsoft Neural (best quality)
- Word buffering for natural sentences
- Adjustable speech rate

## 🧪 Training Your Own Model

### Prepare Dataset
```
data/
├── static/           # For CNN training
│   ├── A/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── B/
│   └── ...
└── dynamic/          # For LSTM training
    ├── hello/
    │   ├── seq001.npy
    │   └── ...
    └── thanks/
```

### Train
```bash
python main.py --mode train --data-dir ./data --output-dir ./outputs
```

### Use Trained Model
```bash
python main.py --mode recognize \
    --static-model outputs/models/static_cnn_best.h5 \
    --dynamic-model outputs/models/dynamic_lstm_best.h5
```

## 📈 Model Architecture Details

### CNN (Static Gestures)
```
Input (64x64x1) → Conv2D(32) → Conv2D(32) → MaxPool → 
Conv2D(64) → Conv2D(64) → MaxPool → 
Conv2D(128) → Conv2D(128) → MaxPool → 
Conv2D(256) → GlobalAvgPool → Dense(512) → Dense(256) → Output(26)
```

### LSTM (Dynamic Gestures)
```
Input (30, 63) → BiLSTM(128) → BiLSTM(128) → BiLSTM(64) → 
Dense(256) → Dense(128) → Output(100)
```

### Hybrid CNN-LSTM
```
Video Input (30, 64, 64, 1) → TimeDistributed(CNN) → 
BiLSTM(128) → BiLSTM(64) → Dense(256) → Output(100)
```

## 🔧 Configuration

```python
# In main.py or config file
config = {
    'confidence_threshold': 0.7,   # Minimum prediction confidence
    'sequence_length': 30,          # Frames for dynamic gestures
    'stability_threshold': 3,       # Consistent predictions needed
    'speech_rate': 150,             # Words per minute
}
```

## 📊 Datasets Used

- **ASL Alphabet Dataset** - 87,000 images
- **WLASL** - Word-Level ASL
- **Custom Collected Data** - 5,000+ sequences

## 🔬 Research & Publication

This system was developed as part of research published in:

> **Journal of Positive School Psychology**
> *"Real-time Sign Language Recognition using Deep Learning for Enhanced Communication Accessibility"*

Key contributions:
- Novel CNN-LSTM hybrid architecture
- 85% accuracy on combined gesture recognition
- Real-time performance (30+ FPS)
- Multi-modal speech output

## 🤝 Accessibility Impact

This system helps:
- Deaf and hard-of-hearing individuals communicate
- Facilitate conversations with hearing people
- Educational tools for ASL learning
- Healthcare communication assistance

## 📜 License

MIT License - Feel free to use for research and accessibility applications.

---

Built with ❤️ for accessibility using TensorFlow, OpenCV, and MediaPipe
