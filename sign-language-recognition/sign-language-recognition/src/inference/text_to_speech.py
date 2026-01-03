"""
Text-to-Speech Conversion Module
Converts recognized gestures to spoken words using multiple TTS engines
"""

import os
import tempfile
from typing import Optional, List
from abc import ABC, abstractmethod
import threading
import queue
import time

# TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import edge_tts
    import asyncio
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class BaseTTS(ABC):
    """Base class for Text-to-Speech engines"""
    
    @abstractmethod
    def speak(self, text: str):
        """Convert text to speech"""
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str):
        """Set the voice for synthesis"""
        pass
    
    @abstractmethod
    def set_rate(self, rate: int):
        """Set speech rate"""
        pass


class Pyttsx3TTS(BaseTTS):
    """
    Offline TTS using pyttsx3
    Works without internet connection
    """
    
    def __init__(self):
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3")
        
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self._setup_default()
    
    def _setup_default(self):
        """Setup default voice settings"""
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Try to use a female voice if available
        for voice in self.voices:
            if 'female' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text: str):
        """Speak the given text"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def speak_async(self, text: str):
        """Speak asynchronously"""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
    
    def set_voice(self, voice_id: str):
        """Set voice by ID"""
        self.engine.setProperty('voice', voice_id)
    
    def set_rate(self, rate: int):
        """Set speech rate (words per minute)"""
        self.engine.setProperty('rate', rate)
    
    def get_available_voices(self) -> List[dict]:
        """Get list of available voices"""
        return [{'id': v.id, 'name': v.name, 'languages': v.languages} 
                for v in self.voices]


class GoogleTTS(BaseTTS):
    """
    Online TTS using Google Text-to-Speech
    Higher quality but requires internet
    """
    
    def __init__(self, lang: str = 'en'):
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS not installed. Run: pip install gtts pygame")
        
        self.lang = lang
        self.slow = False
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
    
    def speak(self, text: str):
        """Speak using Google TTS"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            temp_path = fp.name
        
        try:
            # Generate speech
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(temp_path)
            
            # Play audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def speak_async(self, text: str):
        """Speak asynchronously"""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
    
    def set_voice(self, voice_id: str):
        """Set language (Google TTS doesn't have voice selection)"""
        self.lang = voice_id
    
    def set_rate(self, rate: int):
        """Set speech rate (only slow/normal available)"""
        self.slow = rate < 100


class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS - High quality neural voices
    Requires internet but has excellent voice quality
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural"):
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")
        
        self.voice = voice
        self.rate = "+0%"
        pygame.mixer.init()
    
    async def _speak_async(self, text: str):
        """Internal async speak method"""
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            temp_path = fp.name
        
        try:
            await communicate.save(temp_path)
            
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def speak(self, text: str):
        """Speak using Edge TTS"""
        asyncio.run(self._speak_async(text))
    
    def speak_async(self, text: str):
        """Speak asynchronously"""
        def run():
            asyncio.run(self._speak_async(text))
        threading.Thread(target=run, daemon=True).start()
    
    def set_voice(self, voice_id: str):
        """Set voice by ID"""
        self.voice = voice_id
    
    def set_rate(self, rate: int):
        """Set speech rate (percentage adjustment)"""
        # Convert words per minute to percentage
        adjustment = ((rate - 150) / 150) * 100
        self.rate = f"{adjustment:+.0f}%"
    
    @staticmethod
    async def get_available_voices() -> List[dict]:
        """Get list of available Edge TTS voices"""
        voices = await edge_tts.list_voices()
        return [{'id': v['ShortName'], 'name': v['FriendlyName'], 
                 'locale': v['Locale'], 'gender': v['Gender']} 
                for v in voices]


class SpeechSynthesizer:
    """
    Main speech synthesizer class
    Manages TTS engines and provides unified interface
    """
    
    def __init__(self, engine: str = 'auto'):
        """
        Initialize speech synthesizer
        
        Args:
            engine: 'pyttsx3', 'gtts', 'edge', or 'auto'
        """
        self.tts = self._init_engine(engine)
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self._start_speech_thread()
        
        # Word buffer for sentence formation
        self.word_buffer = []
        self.last_word_time = time.time()
        self.sentence_timeout = 2.0  # seconds
    
    def _init_engine(self, engine: str) -> BaseTTS:
        """Initialize the specified TTS engine"""
        if engine == 'auto':
            # Try engines in order of quality
            if EDGE_TTS_AVAILABLE:
                return EdgeTTS()
            elif GTTS_AVAILABLE:
                return GoogleTTS()
            elif PYTTSX3_AVAILABLE:
                return Pyttsx3TTS()
            else:
                raise RuntimeError("No TTS engine available")
        
        engines = {
            'pyttsx3': Pyttsx3TTS,
            'gtts': GoogleTTS,
            'edge': EdgeTTS
        }
        
        if engine not in engines:
            raise ValueError(f"Unknown engine: {engine}")
        
        return engines[engine]()
    
    def _start_speech_thread(self):
        """Start background thread for speech processing"""
        def speech_worker():
            while True:
                text = self.speech_queue.get()
                if text is None:
                    break
                
                self.is_speaking = True
                try:
                    self.tts.speak(text)
                except Exception as e:
                    print(f"Speech error: {e}")
                finally:
                    self.is_speaking = False
                    self.speech_queue.task_done()
        
        self.speech_thread = threading.Thread(target=speech_worker, daemon=True)
        self.speech_thread.start()
    
    def speak(self, text: str, interrupt: bool = False):
        """
        Speak text
        
        Args:
            text: Text to speak
            interrupt: If True, clear queue and speak immediately
        """
        if interrupt:
            # Clear queue
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
        
        self.speech_queue.put(text)
    
    def speak_word(self, word: str):
        """
        Add word to buffer and speak when sentence is complete
        Uses timeout to detect end of gesture sequence
        """
        current_time = time.time()
        
        # Check if we should speak buffered words (timeout)
        if self.word_buffer and (current_time - self.last_word_time) > self.sentence_timeout:
            sentence = ' '.join(self.word_buffer)
            self.speak(sentence)
            self.word_buffer = []
        
        # Add new word
        self.word_buffer.append(word)
        self.last_word_time = current_time
        
        # Check for sentence-ending punctuation or common endings
        if word.lower() in ['.', '?', '!', 'done', 'end', 'stop']:
            sentence = ' '.join(self.word_buffer)
            self.speak(sentence)
            self.word_buffer = []
    
    def flush_buffer(self):
        """Speak any remaining words in buffer"""
        if self.word_buffer:
            sentence = ' '.join(self.word_buffer)
            self.speak(sentence)
            self.word_buffer = []
    
    def spell_word(self, letters: List[str], delay: float = 0.3):
        """
        Spell out a word letter by letter
        Used for finger spelling
        """
        for letter in letters:
            self.speak(letter)
            time.sleep(delay)
    
    def set_voice(self, voice_id: str):
        """Set TTS voice"""
        self.tts.set_voice(voice_id)
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        self.tts.set_rate(rate)
    
    def stop(self):
        """Stop all speech and clean up"""
        self.speech_queue.put(None)
        self.speech_thread.join(timeout=1)


class GestureToSpeech:
    """
    Converts gesture predictions to natural speech
    Handles word formation, sentence construction, and pronunciation
    """
    
    def __init__(self, synthesizer: Optional[SpeechSynthesizer] = None):
        self.synthesizer = synthesizer or SpeechSynthesizer()
        
        # Common sign language phrases
        self.phrases = {
            'hello': 'Hello!',
            'goodbye': 'Goodbye!',
            'thanks': 'Thank you!',
            'please': 'Please',
            'sorry': 'I am sorry',
            'yes': 'Yes',
            'no': 'No',
            'help': 'I need help',
            'love': 'I love you',
            'understand': 'I understand',
            'dont_understand': "I don't understand",
            'name': 'My name is',
            'how_are_you': 'How are you?',
            'nice_meet': 'Nice to meet you',
        }
        
        # Letter to phonetic mapping for clarity
        self.phonetics = {
            'a': 'ay', 'b': 'bee', 'c': 'see', 'd': 'dee', 'e': 'ee',
            'f': 'ef', 'g': 'jee', 'h': 'aitch', 'i': 'eye', 'j': 'jay',
            'k': 'kay', 'l': 'el', 'm': 'em', 'n': 'en', 'o': 'oh',
            'p': 'pee', 'q': 'cue', 'r': 'ar', 's': 'es', 't': 'tee',
            'u': 'you', 'v': 'vee', 'w': 'double-you', 'x': 'ex',
            'y': 'why', 'z': 'zee'
        }
        
        # Gesture history for context
        self.gesture_history = []
        self.last_gesture = None
        self.last_gesture_time = 0
        self.repeat_threshold = 0.5  # seconds
    
    def process_gesture(self, gesture_label: str, confidence: float = 1.0):
        """
        Process a recognized gesture and convert to speech
        
        Args:
            gesture_label: Predicted gesture class
            confidence: Prediction confidence (0-1)
        """
        current_time = time.time()
        
        # Filter low confidence predictions
        if confidence < 0.7:
            return
        
        # Prevent rapid repeats of same gesture
        if (gesture_label == self.last_gesture and 
            current_time - self.last_gesture_time < self.repeat_threshold):
            return
        
        self.last_gesture = gesture_label
        self.last_gesture_time = current_time
        self.gesture_history.append(gesture_label)
        
        # Check for phrase match
        if gesture_label.lower() in self.phrases:
            text = self.phrases[gesture_label.lower()]
            self.synthesizer.speak(text)
            return
        
        # Check if it's a single letter (finger spelling)
        if len(gesture_label) == 1 and gesture_label.isalpha():
            self.synthesizer.speak_word(gesture_label.upper())
            return
        
        # Default: speak the gesture label
        self.synthesizer.speak_word(gesture_label)
    
    def spell_fingerspelling(self, letters: List[str]):
        """Convert finger-spelled letters to speech"""
        word = ''.join(letters)
        self.synthesizer.speak(word)
    
    def clear_history(self):
        """Clear gesture history"""
        self.gesture_history = []
        self.synthesizer.flush_buffer()
    
    def get_recent_gestures(self, n: int = 10) -> List[str]:
        """Get recent gesture history"""
        return self.gesture_history[-n:]


if __name__ == "__main__":
    # Test TTS
    print("Testing Text-to-Speech...")
    
    synthesizer = SpeechSynthesizer()
    
    # Test basic speech
    print("Testing basic speech...")
    synthesizer.speak("Hello! This is a test of the sign language to speech system.")
    
    time.sleep(3)
    
    # Test word buffer
    print("Testing word buffer...")
    for word in ["Thank", "you", "for", "testing"]:
        synthesizer.speak_word(word)
        time.sleep(0.5)
    
    synthesizer.flush_buffer()
    
    time.sleep(3)
    
    # Test gesture to speech
    print("Testing gesture to speech...")
    g2s = GestureToSpeech(synthesizer)
    
    for gesture in ["hello", "A", "B", "C", "thanks"]:
        g2s.process_gesture(gesture, confidence=0.9)
        time.sleep(1)
    
    print("\nTTS tests completed!")
    synthesizer.stop()
