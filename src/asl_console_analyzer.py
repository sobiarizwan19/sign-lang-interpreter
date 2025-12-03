#!/usr/bin/env python3

import cv2
import os
import tempfile
import sys
from pathlib import Path

# Import the required modules
try:
    from signPredict import ASLClassifier
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("❌ signPredict.py not found!")
    sys.exit(1)

try:
    from predictSentence import SentencePredictor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("❌ predictSentence.py not found!")

class ASLConsoleAnalyzer:
    def __init__(self, 
                 video_path="../content/video.mp4", 
                 frame_gap=10,
                 model_path="../model/retrained_asl_model.pt",
                 gemini_api_key=None,
                 confidence_threshold=0.5,
                 progress_frequency=50):
        """
        Initialize the ASL analyzer
        
        Args:
            video_path: Path to the video file
            frame_gap: Process every Nth frame
            model_path: Path to the ASL model
            gemini_api_key: API key for Gemini
            confidence_threshold: Minimum confidence for including predictions
            progress_frequency: Show progress every N processed frames
        """
        # Convert paths to absolute paths relative to src directory
        src_dir = Path(__file__).parent.absolute()
        
        # Handle video path
        if not os.path.isabs(video_path):
            self.video_path = str(src_dir / video_path)
        else:
            self.video_path = video_path
        
        # Handle model path
        if not os.path.isabs(model_path):
            self.model_path = str(src_dir / model_path)
        else:
            self.model_path = model_path
            
        self.frame_gap = frame_gap
        self.gemini_api_key = gemini_api_key
        self.confidence_threshold = confidence_threshold
        self.progress_frequency = progress_frequency
        
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
                print("🔄 Loading ASL classification model...")
                self.classifier = ASLClassifier(model_path=self.model_path)
                print("✅ ASL model loaded successfully")
            except Exception as e:
                print(f"❌ Model error: {str(e)}")
                self.classifier = None
                sys.exit(1)
        else:
            print("❌ ASL classifier not available")
            sys.exit(1)
    
    def _init_llm(self):
        """Initialize the LLM sentence predictor"""
        if LLM_AVAILABLE:
            try:
                print("🔄 Initializing LLM...")
                if self.gemini_api_key:
                    self.sentence_predictor = SentencePredictor(api_key=self.gemini_api_key)
                else:
                    self.sentence_predictor = SentencePredictor()
                print("✅ LLM initialized successfully")
            except Exception as e:
                print(f"⚠️  LLM initialization error: {e}")
                print("📝 Will only show letter sequence without interpretation")
                self.sentence_predictor = None
        else:
            print("⚠️  LLM not available")
            print("📝 Will only show letter sequence without interpretation")
            self.sentence_predictor = None
    
    def analyze_video(self):
        """
        Analyze the entire video and extract sign language sequence
        """
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            print(f"📁 Current working directory: {os.getcwd()}")
            print(f"📁 Looking for video at: {os.path.abspath(self.video_path)}")
            sys.exit(1)
        
        print(f"🎬 Processing video: {os.path.basename(self.video_path)}")
        print(f"📁 Full path: {self.video_path}")
        print(f"⚙️  Frame gap: {self.frame_gap} (processing every {self.frame_gap} frames)")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print()
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {self.video_path}")
            sys.exit(1)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS")
        print(f"⏱️  Duration: {total_frames/fps:.2f} seconds")
        print()
        
        frame_count = 0
        processed_count = 0
        
        print("🔄 Processing frames...")
        
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
                        print(f"Frame {frame_count:6d}: {pred_letter} ({prediction['confidence']:.1%})")
                
                processed_count += 1
                
                # Show progress
                if processed_count % self.progress_frequency == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📈 Progress: {progress:.1f}% ({processed_count} frames processed)")
            
            frame_count += 1
        
        cap.release()
        
        print()
        print("✅ Video processing complete!")
        print(f"📋 Processed {processed_count} frames out of {total_frames} total frames")
        print(f"🔤 Detected {len(self.sequence)} unique signs")
        print()
        
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
            print(f"⚠️  Prediction error: {e}")
            return None
    
    def _display_results(self):
        """
        Display the final results
        """
        print("=" * 70)
        print("                        FINAL RESULTS")
        print("=" * 70)
        
        if not self.sequence:
            print("❌ No signs detected in the video")
            print(f"💡 Try lowering the confidence threshold (current: {self.confidence_threshold})")
            print(f"💡 Or reducing the frame gap (current: {self.frame_gap})")
            return
        
        # Display detected sequence
        sequence_str = " ".join(self.sequence)
        print(f"🔤 DETECTED LETTER SEQUENCE:")
        print(f"   {sequence_str}")
        print()
        print(f"📊 Detection Statistics:")
        print(f"   • Total unique letters: {len(self.sequence)}")
        print(f"   • Total predicted frames: {len(self.predictions)}")
        print(f"   • Average confidence: {sum(p['confidence'] for p in self.predictions.values()) / len(self.predictions):.1%}")
        print()
        
        # Try to interpret with AI if available
        if self.sentence_predictor:
            print("🤖 AI INTERPRETATION:")
            print("🔄 Analyzing sequence with AI...")
            
            try:
                result = self.sentence_predictor.predict_sentence(sequence_str)
                
                print(f"📝 Original Sequence: {sequence_str}")
                print(f"🎯 Interpretation: {result['interpretation']}")
                print(f"📊 Confidence: {result['confidence']}")
                print(f"💭 Reasoning: {result['reasoning']}")
                
                if result.get('alternatives'):
                    print("🔄 Alternative Interpretations:")
                    for i, alt in enumerate(result['alternatives'], 1):
                        print(f"   {i}. {alt}")
                
            except Exception as e:
                print(f"❌ AI interpretation failed: {e}")
                print("💡 Check your internet connection and API key")
        else:
            print("⚠️  AI interpretation not available")
            print("💡 Install google-generativeai and set GEMINI_API_KEY to enable AI interpretation")
        
        print()
        print("=" * 70)
        print("                     ANALYSIS COMPLETE")
        print("=" * 70)
        
        # Show configuration summary
        print()
        print("📋 Configuration used:")
        print(f"   • Video: {os.path.basename(self.video_path)}")
        print(f"   • Frame gap: {self.frame_gap}")
        print(f"   • Confidence threshold: {self.confidence_threshold}")
        print(f"   • Model: {os.path.basename(self.model_path)}")
        print(f"   • AI enabled: {'Yes' if self.sentence_predictor else 'No'}")