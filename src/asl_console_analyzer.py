#!/usr/bin/env python3

import cv2
import os
import tempfile
import sys
from pathlib import Path

# Import the required modules - adjusted for Flask context
try:
    from signPredict import ASLClassifier
    MODEL_AVAILABLE = True
except ImportError:
    try:
        from src.signPredict import ASLClassifier
        MODEL_AVAILABLE = True
    except ImportError:
        MODEL_AVAILABLE = False
        sys.exit(1)

try:
    from predictSentence import SentencePredictor
    LLM_AVAILABLE = True
except ImportError:
    try:
        from src.predictSentence import SentencePredictor
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False

class ASLConsoleAnalyzer:
    def __init__(self, 
                 video_path="../content/video.mp4", 
                 frame_gap=10,
                 model_path="../model/retrained_asl_model.pt",
                 gemini_api_key=None,
                 gemini_model=None,
                 confidence_threshold=0.5):
        """
        Initialize the ASL analyzer
        
        Args:
            video_path: Path to the video file
            frame_gap: Process every Nth frame
            model_path: Path to the ASL model
            gemini_api_key: API key for Gemini
            gemini_model: Gemini model to use
            confidence_threshold: Minimum confidence for including predictions
        """
        # Convert paths to absolute paths relative to current directory (not src)
        current_dir = Path(__file__).parent.absolute()
        
        # Handle video path
        if not os.path.isabs(video_path):
            self.video_path = str(current_dir / video_path)
        else:
            self.video_path = video_path
        
        # Handle model path
        if not os.path.isabs(model_path):
            self.model_path = str(current_dir / model_path)
        else:
            self.model_path = model_path
            
        self.frame_gap = frame_gap
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.confidence_threshold = confidence_threshold
        
        self.classifier = None
        self.sentence_predictor = None
        self.predictions = {}
        self.sequence = []
        
        # Initialize components
        self._init_classifier()
        self._init_llm()
        
    def _init_classifier(self):
        """Initialize the sign language classifier"""
        if MODEL_AVAILABLE:
            try:
                self.classifier = ASLClassifier(model_path=self.model_path)
            except Exception as e:
                self.classifier = None
                raise Exception(f"Failed to initialize classifier: {e}")
        else:
            raise Exception("ASL classifier not available")
    
    def _init_llm(self):
        """Initialize the LLM sentence predictor"""
        if LLM_AVAILABLE:
            try:
                if self.gemini_api_key:
                    self.sentence_predictor = SentencePredictor(
                        api_key=self.gemini_api_key,
                        model_name=self.gemini_model
                    )
                else:
                    self.sentence_predictor = SentencePredictor(model_name=self.gemini_model)
            except Exception as e:
                self.sentence_predictor = None
        else:
            self.sentence_predictor = None
    
    def analyze_video(self):
        """
        Analyze the entire video and extract sign language sequence
        """
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {self.video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every frame_gap frames
            if frame_count % self.frame_gap == 0:
                prediction = self._predict_frame(frame)
                if prediction and prediction['confidence'] > self.confidence_threshold:
                    self.predictions[frame_count] = prediction
                    
                    # Add to sequence if it's different from the last prediction
                    pred_letter = prediction['prediction']
                    if not self.sequence or self.sequence[-1] != pred_letter:
                        self.sequence.append(pred_letter)
            
            frame_count += 1
        
        cap.release()
        
        # Display results
        self._display_results()
    
    def _predict_frame(self, frame):
        """
        Predict sign for a frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            dict: Prediction result
        """
        if self.classifier is None:
            return None
        
        try:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            # Get prediction
            result = self.classifier.predict_single_image(temp_path, show=False, save=False)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return {
                "prediction": result['top_class'],
                "confidence": float(result['top_confidence']),
                "all_predictions": result['top5']
            }
            
        except Exception as e:
            return None
    
    def _display_results(self):
        """
        Display the final results
        """
        if not self.sequence:
            print("No signs detected in the video")
            return
        
        # Display detected sequence
        sequence_str = " ".join(self.sequence)
        print(f"DETECTED SEQUENCE: {sequence_str}")
        
        # Try to interpret with AI if available
        if self.sentence_predictor:
            try:
                result = self.sentence_predictor.predict_sentence(sequence_str)
                print(f"INTERPRETATION: {result['interpretation']}")
                
            except Exception as e:
                print(f"INTERPRETATION: {sequence_str}")
        else:
            print(f"INTERPRETATION: {sequence_str}")