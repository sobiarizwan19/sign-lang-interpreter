#!/usr/bin/env python3
"""
ASL Video Analyzer with Frequency Analysis
Complete project in a single file
"""

import cv2
import os
import tempfile
import sys
import math
from pathlib import Path
import asyncio
import numpy as np
from nicegui import ui, app
from dotenv import load_dotenv
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image

# Try to import MediaPipe, but make it optional
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("⚠ Warning: MediaPipe not installed. Background removal will be disabled.")
    MEDIAPIPE_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================

# Load environment variables
load_dotenv()

# Configuration variables
VIDEO_PATH = os.getenv('VIDEO_PATH', '../content/demo.mp4')
FRAME_GAP = int(os.getenv('FRAME_GAP', '10'))
MODEL_PATH = os.getenv('MODEL_PATH', '../model/retrained_asl_model.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
OUTLIER_STD_THRESHOLD = float(os.getenv('OUTLIER_STD_THRESHOLD', '1.5'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
ENABLE_PREPROCESSING = os.getenv('ENABLE_PREPROCESSING', 'true').lower() == 'true'

# ============================================================================
# Video Preprocessor (MediaPipe Background Removal)
# ============================================================================

class VideoPreprocessor:
    """Handles video preprocessing including background removal"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        
    def remove_background_mediapipe_fast(self, input_path, output_path, start_frame=None, end_frame=None):
        """
        Fast hand segmentation using MediaPipe
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            start_frame: Optional start frame (if None, process from beginning)
            end_frame: Optional end frame (if None, process to end)
            
        Returns:
            tuple: (start_frame, end_frame, processed_count) or None if failed
        """
        if not self.mediapipe_available:
            print("❌ MediaPipe not available. Skipping background removal.")
            return None
        
        print("Initializing MediaPipe...")
        
        try:
            # Initialize MediaPipe Selfie Segmentation
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
            
            print("Opening video...")
            cap = cv2.VideoCapture(input_path)
            
            if not cap.isOpened():
                print(f"❌ Cannot open video: {input_path}")
                return None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\n📹 Video Information:")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Total frames: {total_frames}")
            print(f"  Duration: {total_frames/fps:.2f} seconds")
            
            # Determine frame range
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = total_frames - 1
            
            # Validate frame range
            if start_frame < 0 or start_frame >= total_frames:
                start_frame = 0
            if end_frame <= start_frame or end_frame > total_frames:
                end_frame = total_frames - 1
            
            # Calculate number of frames to process
            frames_to_process = end_frame - start_frame + 1
            
            print(f"\n🔄 Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
            print(f"  Segment duration: {frames_to_process/fps:.2f} seconds")
            
            # Setup output - we'll write at the original fps
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            processed_count = 0
            print("\n⚡ Processing started...")
            
            while cap.isOpened() and processed_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = start_frame + processed_count
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = selfie_segmentation.process(rgb_frame)
                
                # Get segmentation mask
                if results.segmentation_mask is not None:
                    # Create mask - adjust threshold as needed
                    mask = results.segmentation_mask > 0.1
                    
                    # Create black background
                    black_bg = np.zeros_like(frame)
                    
                    # Apply mask
                    for c in range(3):
                        black_bg[:, :, c] = np.where(
                            mask == 1, 
                            frame[:, :, c], 
                            0  # Black background
                        )
                    
                    out.write(black_bg)
                else:
                    # If no mask, write black frame
                    out.write(np.zeros_like(frame))
                
                processed_count += 1
                
                # Show progress every 10 frames
                if processed_count % 10 == 0:
                    progress_pct = processed_count/frames_to_process*100
                    print(f"  Processed frame {current_frame}/{end_frame} "
                          f"({processed_count}/{frames_to_process} frames | "
                          f"{progress_pct:.1f}%)")
            
            cap.release()
            out.release()
            selfie_segmentation.close()
            
            print(f"\n✅ Preprocessing complete!")
            print(f"  Processed {processed_count} frames ({start_frame} to {end_frame})")
            print(f"  Output saved to: {output_path}")
            
            # Return the frame range for reference
            return start_frame, end_frame, processed_count
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_video(self, input_path, output_dir="preprocessed", frame_range=None):
        """
        Preprocess video by removing background
        
        Args:
            input_path: Path to input video
            output_dir: Directory to save preprocessed video
            frame_range: Optional tuple (start_frame, end_frame)
            
        Returns:
            str: Path to preprocessed video or original if preprocessing fails
        """
        if not ENABLE_PREPROCESSING or not self.mediapipe_available:
            print("⚠ Preprocessing disabled or MediaPipe not available")
            return input_path
        
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Generate output filename
            input_name = Path(input_path).stem
            output_path = str(Path(output_dir) / f"{input_name}_preprocessed.mp4")
            
            print(f"\n🔧 Starting video preprocessing...")
            print(f"  Input: {input_path}")
            print(f"  Output: {output_path}")
            
            # Extract frame range if provided
            start_frame = None
            end_frame = None
            if frame_range and len(frame_range) == 2:
                start_frame, end_frame = frame_range
            
            # Process video
            result = self.remove_background_mediapipe_fast(
                input_path, output_path, start_frame, end_frame
            )
            
            if result:
                print(f"✅ Successfully preprocessed video")
                return output_path
            else:
                print(f"⚠ Preprocessing failed, using original video")
                return input_path
                
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return input_path
    
    def extract_video_info(self, video_path):
        """Extract basic information from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'duration': duration,
                'resolution': f"{width}x{height}"
            }
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return None

# ============================================================================
# ASL Classifier (YOLO Model)
# ============================================================================

class ASLClassifier:
    """
    ASL Sign Language Classifier using trained YOLO model
    """
    def __init__(self, model_path=None):
        """
        Initialize the classifier with a trained model
        Args:
            model_path: Path to the trained model weights
        """
        # Default model paths to try
        current_dir = Path(__file__).parent.absolute()
        if model_path:
            possible_paths = [model_path]
        else:
            possible_paths = [
                str(current_dir / MODEL_PATH),
                str(current_dir / "../model/retrained_asl_model.pt"),
                str(current_dir / "../../model/retrained_asl_model.pt"),
                "./model/retrained_asl_model.pt",
                "../model/retrained_asl_model.pt",
                "model/retrained_asl_model.pt",
                "./retrained_asl_model.pt",
                "retrained_asl_model.pt"
            ]

        self.model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                self.model_path = abs_path
                break

        if self.model_path is None:
            raise FileNotFoundError(
                f"ASL model not found. Searched paths:\n" +
                "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths) +
                f"\n\nCurrent working directory: {os.getcwd()}"
            )

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def predict_single_image(self, image_path, show=False, save=False, save_path="./prediction.jpg"):
        """
        Classify a single image
        Args:
            image_path: Path to the image file
            show: Whether to display the result
            save: Whether to save the result
            save_path: Path to save the result
        Returns:
            dict: Prediction results with top predictions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Run prediction
        results = self.model(image_path)
        result = results[0]

        # Extract predictions
        probs = result.probs
        top5_indices = probs.top5
        top5_conf = probs.top5conf.tolist()
        class_names = result.names

        # Build prediction dictionary
        predictions = {
            'top_class': class_names[top5_indices[0]],
            'top_confidence': top5_conf[0],
            'top5': [
                {
                    'class': class_names[idx],
                    'confidence': conf
                }
                for idx, conf in zip(top5_indices, top5_conf)
            ]
        }

        return predictions

    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'model_name': os.path.basename(self.model_path),
            'classes': list(self.model.names.values()) if hasattr(self.model, 'names') else None
        }

# ============================================================================
# Sentence Predictor (Gemini LLM)
# ============================================================================

class SentencePredictor:
    """
    Predicts sentences from ASL alphabet sequences using Google Gemini
    with support for weighted frequency analysis
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini API
        
        Args:
            api_key: Google Gemini API key (if None, uses default)
            model_name: Gemini model to use (if None, tries defaults)
        """
        # Use provided API key or default
        if api_key is None:
            api_key = GEMINI_API_KEY
        
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        
        # Use provided model or try defaults
        if model_name:
            # Remove 'models/' prefix if present for SDK initialization
            clean_model_name = model_name.replace('models/', '')
            model_attempts = [clean_model_name]
        else:
            model_attempts = [
                'gemini-2.0-flash-exp',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
        
        self.model = None
        self.model_name = None
        
        # Safety settings to prevent blocking
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        for model_name in model_attempts:
            try:
                test_model = genai.GenerativeModel(
                    model_name,
                    safety_settings=safety_settings
                )
                # Test with simple prompt
                test_response = test_model.generate_content("Test")
                # Verify we can access text
                _ = test_response.text
                self.model = test_model
                self.model_name = model_name
                print(f"✓ Successfully initialized model: {model_name}")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"✗ Failed to initialize {model_name}: {error_msg[:100]}")
                if "API key" in error_msg.lower():
                    break  # Don't try other models if API key is invalid
                continue
        
        if self.model is None:
            raise Exception("Could not initialize any Gemini model")
    
    def predict_weighted_sentence(self, frequency_sequence: str) -> dict:
        """
        Predict sentence from weighted frequency sequence
        
        Args:
            frequency_sequence: String of space-separated letter:count pairs (e.g., "L:6 O:4 V:5 E:6")
            
        Returns:
            dict with interpretation results
        """
        if not frequency_sequence or frequency_sequence.strip() == "":
            return {
                'interpretation': "No sequence provided",
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': "Empty sequence",
                'original_sequence': ""
            }
        
        if self.model is None:
            return {
                'interpretation': "Model not initialized",
                'alternatives': [],
                'confidence': "LOW", 
                'raw_response': "",
                'reasoning': "Gemini model failed to initialize",
                'original_sequence': frequency_sequence
            }
        
        prompt = self._create_weighted_prompt(frequency_sequence)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Lower temperature for more consistent results
                    max_output_tokens=500,
                )
            )
            
            # Check if response was blocked
            if not response.parts:
                return {
                    'interpretation': self._sequence_to_text(frequency_sequence),
                    'alternatives': [],
                    'confidence': "LOW",
                    'raw_response': "",
                    'reasoning': "Response blocked by safety filters",
                    'original_sequence': frequency_sequence
                }
            
            # Try to get text
            try:
                response_text = response.text
            except Exception as e:
                # If response.text fails, try to get from parts
                if response.parts:
                    response_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
                else:
                    raise e
            
            result = self._parse_weighted_response(response_text, frequency_sequence)
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"API Error: {error_msg}")
            
            # Fallback: return simple interpretation
            simple_interpretation = self._sequence_to_text(frequency_sequence)
            return {
                'interpretation': simple_interpretation,
                'alternatives': [],
                'confidence': "LOW",
                'raw_response': "",
                'reasoning': f"API error - showing raw sequence. Error: {error_msg[:100]}",
                'original_sequence': frequency_sequence
            }
    
    def _create_weighted_prompt(self, frequency_sequence: str) -> str:
        """Create prompt for weighted frequency interpretation"""
        
        # Parse the frequency sequence for display
        pairs = []
        for item in frequency_sequence.split():
            if ':' in item:
                letter, count = item.split(':')
                pairs.append(f"({letter},{count})")
            else:
                pairs.append(f"({item},1)")
        
        pairs_str = ", ".join(pairs)
        
        prompt = f"""You are an expert ASL (American Sign Language) fingerspelling interpreter. Your task is to interpret a WEIGHTED sequence of letters from ASL fingerspelling.

WEIGHTED FREQUENCY SEQUENCE: {pairs_str}

Each pair (letter, count) represents:
- Letter: The detected ASL letter
- Count: How many consecutive frames showed this letter (weight/importance)

IMPORTANT RULES FOR INTERPRETATION:
1. HIGHER COUNT = MORE IMPORTANT: Letters with higher counts are more reliable and should be given more weight in the interpretation.

2. NOISE HANDLING:
   - Low-count letters (1-2) might be noise or transition artifacts
   - Medium-count letters (3-5) are likely real but could have some noise
   - High-count letters (6+) are almost certainly intentional signs

3. WORD FORMATION PRIORITIES:
   a) Use high-count letters as anchors for the word/phrase
   b) Medium-count letters fill in between anchors
   c) Low-count letters are optional - include only if they make sense

4. COMMON PATTERNS TO CONSIDER:
   - Repeated letters often indicate emphasis or part of common words (e.g., "LL" in "HELLO", "OO" in "GOOD")
   - Consider that some signs might be held longer (higher count) for emphasis or clarity

5. YOUR OUTPUT SHOULD:
   - Form a coherent English word, phrase, or sentence
   - Respect the weight/importance indicated by the counts
   - Ignore clearly erroneous patterns
   - Consider common names, words, and phrases

EXAMPLE INTERPRETATIONS:
- (H,1),(E,4),(L,6),(L,2),(O,5) → "HELLO" (L and O have high weights, E is medium, H is low but makes sense)
- (T,2),(H,3),(A,5),(N,4),(K,6),(Y,2),(O,3),(U,5) → "THANK YOU"
- (M,3),(Y,4),(N,5),(A,6),(M,4),(E,5) → "MY NAME"
- (C,6),(O,5),(F,4),(F,3),(E,5),(E,2) → "COFFEE"
- (S,5),(T,4),(A,6),(N,5),(F,3),(O,4),(R,4),(D,5) → "STANFORD"

YOUR TASK: Given the weighted frequency sequence above, provide:
1. The most likely English interpretation
2. Brief reasoning for your choice

Interpretation:"""
        return prompt
    
    def _parse_weighted_response(self, response_text: str, original_sequence: str) -> dict:
        """Parse Gemini's weighted response into structured format"""
        # Clean the response text
        lines = response_text.strip().split('\n')
        
        interpretation = ""
        reasoning = ""
        
        # Try to extract interpretation and reasoning
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for interpretation markers
            if any(marker in line.lower() for marker in ['interpretation:', 'output:', 'result:', 'word:', 'phrase:']):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    interpretation = parts[1].strip().strip('"\'')
            elif not interpretation and i == 0:
                # First non-empty line might be the interpretation
                interpretation = line.strip('"\'').split('.')[0]
            elif 'reasoning' in line.lower() or 'because' in line.lower() or len(line) > 50:
                reasoning = line
        
        # If no interpretation found, use the whole response
        if not interpretation:
            interpretation = lines[0].strip().strip('"\'').split('.')[0]
        
        # If no reasoning found, create simple reasoning
        if not reasoning:
            reasoning = "Based on weighted frequency analysis of ASL signs"
        
        # Parse the original sequence to calculate statistics
        total_weight = 0
        letter_weights = {}
        
        for item in original_sequence.split():
            if ':' in item:
                letter, count = item.split(':')
                weight = int(count)
                total_weight += weight
                if letter in letter_weights:
                    letter_weights[letter] += weight
                else:
                    letter_weights[letter] = weight
        
        # Calculate confidence based on weight distribution
        confidence = "MEDIUM"
        if total_weight > 0:
            # High confidence if most weight is concentrated in few letters
            sorted_weights = sorted(letter_weights.values(), reverse=True)
            if len(sorted_weights) >= 2:
                top2_ratio = sum(sorted_weights[:2]) / total_weight
                if top2_ratio > 0.6:
                    confidence = "HIGH"
                elif top2_ratio < 0.3:
                    confidence = "LOW"
        
        # Generate alternatives
        alternatives = []
        simple_text = self._sequence_to_text(original_sequence)
        if simple_text != interpretation:
            alternatives.append(simple_text)
        
        result = {
            'interpretation': interpretation,
            'alternatives': alternatives,
            'confidence': confidence,
            'raw_response': response_text,
            'reasoning': reasoning,
            'original_sequence': original_sequence,
            'total_weight': total_weight,
            'letter_weights': letter_weights
        }
        
        return result
    
    def _sequence_to_text(self, frequency_sequence: str) -> str:
        """Convert frequency sequence to simple text"""
        letters = []
        for item in frequency_sequence.split():
            if ':' in item:
                letter, count = item.split(':')
                # Repeat letter based on weight (but cap at reasonable amount)
                repeat = min(int(count), 3)  # Cap at 3 repeats
                letters.append(letter * repeat)
            else:
                letters.append(item)
        
        return ''.join(letters)
    
    def predict_sentence(self, alphabet_sequence: str) -> dict:
        """
        Original method - for backward compatibility
        """
        # Convert to weighted format and call the new method
        weighted_seq = " ".join([f"{letter}:1" for letter in alphabet_sequence.split()])
        return self.predict_weighted_sentence(weighted_seq)
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'api_key_preview': f"{self.api_key[:10]}..." if self.api_key else None,
            'available': self.model is not None
        }

# ============================================================================
# Main ASL Analyzer
# ============================================================================

class ASLAnalyzer:
    def __init__(self, 
                 video_path="../content/video.mp4", 
                 frame_gap=10,
                 model_path="../model/retrained_asl_model.pt",
                 gemini_api_key=None,
                 gemini_model=None,
                 confidence_threshold=0.5,
                 outlier_std_threshold=1.5,
                 enable_preprocessing=True):
        """
        Initialize the ASL analyzer
        
        Args:
            video_path: Path to the video file
            frame_gap: Process every Nth frame
            model_path: Path to the ASL model
            gemini_api_key: API key for Gemini
            gemini_model: Gemini model to use
            confidence_threshold: Minimum confidence for including predictions
            outlier_std_threshold: Standard deviation threshold for outlier removal
            enable_preprocessing: Whether to preprocess videos with background removal
        """
        # Convert paths to absolute paths
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
        self.outlier_std_threshold = outlier_std_threshold
        self.enable_preprocessing = enable_preprocessing
        
        self.classifier = None
        self.sentence_predictor = None
        self.preprocessor = None
        self.predictions = {}
        self.raw_predictions = []
        self.frequency_sequence = []
        
        # Initialize components
        self._init_preprocessor()
        self._init_classifier()
        self._init_llm()
        
    def _init_preprocessor(self):
        """Initialize the video preprocessor"""
        try:
            self.preprocessor = VideoPreprocessor()
            if self.preprocessor.mediapipe_available:
                print("✓ Video preprocessor initialized (MediaPipe available)")
            else:
                print("⚠ Video preprocessor initialized (MediaPipe not available)")
        except Exception as e:
            self.preprocessor = None
            print(f"⚠ Failed to initialize preprocessor: {e}")
    
    def _init_classifier(self):
        """Initialize the sign language classifier"""
        try:
            self.classifier = ASLClassifier(model_path=self.model_path)
            print(f"✓ Classifier loaded: {self.classifier.model_path}")
        except Exception as e:
            self.classifier = None
            raise Exception(f"Failed to initialize classifier: {e}")
    
    def _init_llm(self):
        """Initialize the LLM sentence predictor"""
        try:
            if self.gemini_api_key:
                self.sentence_predictor = SentencePredictor(
                    api_key=self.gemini_api_key,
                    model_name=self.gemini_model
                )
            else:
                self.sentence_predictor = SentencePredictor(model_name=self.gemini_model)
            print(f"✓ Gemini predictor initialized: {self.sentence_predictor.model_name}")
        except Exception as e:
            self.sentence_predictor = None
            print(f"⚠ Warning: Could not initialize LLM: {e}")
    
    def preprocess_video(self, video_path=None):
        """
        Preprocess the video (background removal, etc.)
        
        Args:
            video_path: Path to video to preprocess (if None, uses self.video_path)
            
        Returns:
            str: Path to preprocessed video
        """
        if not self.enable_preprocessing or not self.preprocessor:
            print("⚠ Preprocessing disabled or preprocessor not available")
            return self.video_path if video_path is None else video_path
        
        input_path = video_path if video_path is not None else self.video_path
        
        if not os.path.exists(input_path):
            print(f"❌ Video file not found: {input_path}")
            return input_path
        
        print(f"\n🔧 Starting video preprocessing...")
        video_info = self.preprocessor.extract_video_info(input_path)
        if video_info:
            print(f"  Original video: {video_info['resolution']}, {video_info['duration']:.2f}s")
        
        # Preprocess the video
        preprocessed_path = self.preprocessor.preprocess_video(input_path)
        
        # Update video path if preprocessing was successful
        if preprocessed_path != input_path:
            print(f"✅ Using preprocessed video: {preprocessed_path}")
            if video_path is None:
                self.video_path = preprocessed_path
            return preprocessed_path
        else:
            print(f"⚠ Using original video (preprocessing skipped)")
            return input_path
    
    def analyze_video(self, progress_callback=None, preprocess=True):
        """
        Analyze the entire video and extract sign language sequence using frequency analysis
        
        Args:
            progress_callback: Optional callback for progress updates
            preprocess: Whether to preprocess the video first
            
        Returns:
            dict: Analysis results
        """
        # Preprocess video if enabled
        if preprocess and self.enable_preprocessing:
            self.preprocess_video()
        
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"❌ Video file not found: {self.video_path}")
            return {
                'error': f'Video file not found: {self.video_path}',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {self.video_path}")
            return {
                'error': f'Cannot open video: {self.video_path}',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        raw_predictions_list = []
        
        print(f"\n📊 Processing video: {total_frames} frames total")
        print(f"  Processing every {self.frame_gap} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every frame_gap frames
            if frame_count % self.frame_gap == 0:
                prediction = self._predict_frame(frame)
                if prediction:
                    pred_letter = prediction['prediction']
                    raw_predictions_list.append(pred_letter)
                    
                    # Store for debugging
                    self.predictions[frame_count] = prediction
            
            frame_count += 1
            
            # Progress callback
            if progress_callback and frame_count % (self.frame_gap * 10) == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
        
        cap.release()
        
        if not raw_predictions_list:
            print("❌ No signs detected in the video")
            return {
                'error': 'No signs detected in video',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        self.raw_predictions = raw_predictions_list
        
        print(f"\n📝 RAW PREDICTIONS: {''.join(self.raw_predictions)}")
        print(f"  Raw predictions count: {len(self.raw_predictions)}")
        
        # Process frequency analysis
        self._process_frequency_analysis()
        
        # Get interpretation
        return self._get_interpretation()
    
    def _process_frequency_analysis(self):
        """Process raw predictions - SIMPLE VERSION (no outlier removal)"""
        if not self.raw_predictions:
            return
        
        print("\n📊 === FREQUENCY ANALYSIS ===")
        
        # Step 1: Group consecutive identical letters and count them
        groups = []
        current_letter = self.raw_predictions[0]
        current_count = 1
        
        for i in range(1, len(self.raw_predictions)):
            if self.raw_predictions[i] == current_letter:
                current_count += 1
            else:
                groups.append((current_letter, current_count))
                current_letter = self.raw_predictions[i]
                current_count = 1
        
        # Add the last group
        groups.append((current_letter, current_count))
        
        print(f"  Grouped predictions: {groups}")
        
        # Step 2: NO OUTLIER REMOVAL - keep all groups
        print("  No outlier removal - keeping all detected sequences")
        
        # Step 3: Merge consecutive same letters ONLY
        # (This handles cases like (G,18), (C,1), (G,2) -> (G,20), (C,1))
        merged_groups = []
        if groups:
            current_letter, current_total = groups[0]
            
            for i in range(1, len(groups)):
                letter, count = groups[i]
                if letter == current_letter:
                    # Merge consecutive same letters
                    current_total += count
                else:
                    merged_groups.append((current_letter, current_total))
                    current_letter = letter
                    current_total = count
            
            # Add the last merged group
            merged_groups.append((current_letter, current_total))
        
        self.frequency_sequence = merged_groups
        print(f"\n✅ Final frequency sequence (after merging): {self.frequency_sequence}")
        
        # Verify by reconstructing
        reconstructed = ''.join([letter * count for letter, count in merged_groups])
        original = ''.join(self.raw_predictions)
        print(f"\n🔍 Verification:")
        print(f"  Original length: {len(original)}")
        print(f"  Reconstructed length: {len(reconstructed)}")
        print(f"  Match: {original == reconstructed}")
    
    def _predict_frame(self, frame):
        """
        Predict sign for a frame - ONLY ACCEPT ALPHABET CHARACTERS (A-Z)
        If no alphabet detected above threshold, ignore the frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            dict: Prediction result or None if not alphabet
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
            
            # Extract prediction and confidence
            pred_letter = result['top_class']
            confidence = float(result['top_confidence'])
            
            # Define valid ASL alphabet letters (ONLY A-Z)
            valid_asl_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            # Check if it's a valid ASL alphabet character (A-Z) and above confidence threshold
            if (len(pred_letter) == 1 and 
                pred_letter.upper() in valid_asl_letters and 
                confidence > self.confidence_threshold):
                
                # Convert to uppercase for consistency
                pred_letter = pred_letter.upper()
                
                return {
                    "prediction": pred_letter,
                    "confidence": confidence,
                    "all_predictions": result['top5']
                }
            else:
                # Ignore this frame - not a valid ASL alphabet or below threshold
                return None
                
        except Exception as e:
            return None
    
    def _get_interpretation(self):
        """
        Get AI interpretation of the frequency sequence
        """
        if not self.frequency_sequence:
            return {
                'error': 'No frequency sequence generated',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        # Display frequency sequence
        sequence_str = " ".join([f"{letter}" for letter, _ in self.frequency_sequence])
        weighted_str = ", ".join([f"({letter},{count})" for letter, count in self.frequency_sequence])
        freq_str = " ".join([f"{letter}:{count}" for letter, count in self.frequency_sequence])
        
        print(f"\n🎯 === FINAL RESULTS ===")
        print(f"  RAW LETTERS: {''.join(self.raw_predictions)}")
        print(f"  FREQUENCY SEQUENCE: {weighted_str}")
        print(f"  SIMPLE SEQUENCE: {sequence_str}")
        
        # Try to interpret with AI if available
        if self.sentence_predictor:
            try:
                result = self.sentence_predictor.predict_weighted_sentence(freq_str)
                
                return {
                    'raw_sequence': ''.join(self.raw_predictions),
                    'frequency_sequence': freq_str,
                    'simple_sequence': sequence_str,
                    'interpretation': result['interpretation'],
                    'confidence': result['confidence'],
                    'reasoning': result.get('reasoning', ''),
                    'alternatives': result.get('alternatives', []),
                    'frequency_groups': self.frequency_sequence,
                    'preprocessed': self.enable_preprocessing
                }
                
            except Exception as e:
                print(f"❌ Error in AI interpretation: {e}")
                return {
                    'raw_sequence': ''.join(self.raw_predictions),
                    'frequency_sequence': freq_str,
                    'simple_sequence': sequence_str,
                    'interpretation': sequence_str,
                    'confidence': 'LOW',
                    'reasoning': f'AI interpretation failed: {str(e)}',
                    'alternatives': [],
                    'frequency_groups': self.frequency_sequence,
                    'preprocessed': self.enable_preprocessing
                }
        else:
            return {
                'raw_sequence': ''.join(self.raw_predictions),
                'frequency_sequence': freq_str,
                'simple_sequence': sequence_str,
                'interpretation': sequence_str,
                'confidence': 'N/A',
                'reasoning': 'No LLM available for interpretation',
                'alternatives': [],
                'frequency_groups': self.frequency_sequence,
                'preprocessed': self.enable_preprocessing
            }

# ============================================================================
# Web Application
# ============================================================================

# Global objects
classifier = None
sentence_predictor = None
video_preprocessor = None

def init_models():
    """Initialize the ASL classifier and sentence predictor"""
    global classifier, sentence_predictor, video_preprocessor
    
    try:
        # Initialize video preprocessor
        video_preprocessor = VideoPreprocessor()
        if video_preprocessor.mediapipe_available:
            print(f"✓ Video preprocessor initialized (MediaPipe available)")
        else:
            print(f"⚠ Video preprocessor initialized (MediaPipe not available)")
        
        # Initialize classifier
        classifier = ASLClassifier(model_path=MODEL_PATH)
        print(f"✓ Classifier loaded: {classifier.model_path}")
        
        # Initialize sentence predictor with Gemini
        if GEMINI_API_KEY:
            sentence_predictor = SentencePredictor(
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL
            )
            print(f"✓ Gemini predictor initialized: {sentence_predictor.model_name}")
        else:
            print("⚠ Warning: GEMINI_API_KEY not found. AI interpretation will be limited.")
            sentence_predictor = None
            
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        raise

def process_video_with_frequency(video_path: str, progress_callback=None, preprocess=True):
    """
    Process video and extract ASL sequence with frequency analysis
    
    Args:
        video_path: Path to the video file
        progress_callback: Optional callback for progress updates
        preprocess: Whether to preprocess the video first
        
    Returns:
        dict: Results including sequence and interpretation
    """
    # Create analyzer instance
    analyzer = ASLAnalyzer(
        video_path=video_path,
        frame_gap=FRAME_GAP,
        model_path=MODEL_PATH,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        outlier_std_threshold=OUTLIER_STD_THRESHOLD,
        enable_preprocessing=ENABLE_PREPROCESSING
    )
    
    # Analyze video
    return analyzer.analyze_video(progress_callback, preprocess)

# Custom CSS for clean design
modern_css = """
/* Clean gradient background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Upload area styling */
.upload-area {
    border: 3px dashed rgba(255, 255, 255, 0.4);
    transition: all 0.3s ease;
    border-radius: 16px;
    padding: 60px 40px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    cursor: pointer;
}

.upload-area:hover {
    border-color: rgba(255, 255, 255, 0.7);
    background: rgba(255, 255, 255, 0.15);
}

/* Progress bar styling */
.progress-bar {
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.2);
}

.progress-bar .q-linear-progress__track {
    background: linear-gradient(90deg, #4ade80, #3b82f6) !important;
    border-radius: 3px;
}

/* Output area styling */
.output-area {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    min-height: 120px;
    padding: 30px;
    animation: fadeIn 0.5s ease-out;
}

/* Preprocessing info styling */
.preprocessing-info {
    background: rgba(59, 130, 246, 0.1);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    border-left: 4px solid #3b82f6;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Typography */
.ai-output-text {
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1.4;
    color: #1f2937;
    text-align: center;
    margin: 0;
}
"""

@ui.page('/')
async def main_page():
    """Minimal page with only drop box and output"""
    
    # Inject custom CSS
    ui.add_head_html(f'<style>{modern_css}</style>')
    
    # State variables
    processing = {'active': False}
    
    # Main container
    with ui.column().classes('w-full min-h-screen items-center justify-center p-4 md:p-8 gap-8'):
        
        # Drop Box Area
        with ui.column().classes('w-full max-w-2xl items-center gap-4'):
            # Progress indicator (hidden by default)
            progress_container = ui.column().classes('w-full items-center gap-3')
            with progress_container:
                progress_bar = ui.linear_progress(value=0).classes('progress-bar w-full')
                progress_bar.visible = False
                status_label = ui.label('').classes('text-white/80 text-sm')
                status_label.visible = False
            
            # Preprocessing info - SIMPLIFIED: just show if preprocessing is enabled
            if ENABLE_PREPROCESSING and MEDIAPIPE_AVAILABLE:
                with ui.column().classes('w-full preprocessing-info'):
                    ui.label('🛠️ Video Preprocessing Enabled').classes('text-blue-600 font-medium text-sm')
                    ui.label('Background will be automatically removed').classes('text-blue-500/80 text-xs')
            
            # File upload area - Drop Box
            ui.upload(
                on_upload=lambda e: handle_upload(e, progress_bar, status_label),
                max_files=1,
                auto_upload=True
            ).props('''
                accept="video/*"
                label="📁 Drop ASL video here or click to browse"
                color="white"
                text-color="white"
            ''').classes('upload-area w-full').style('width: 100%')
        
        # Output Area (hidden initially)
        output_area = ui.column().classes('output-area w-full max-w-3xl')
        output_area.visible = False
        
        # Initialize output container
        with output_area:
            output_container = ui.column().classes('w-full items-center justify-center')
        
        async def handle_upload(e, progress_bar, status_label):
            """Handle video upload and processing with frequency analysis"""
            if processing['active']:
                return
            
            try:
                # NiceGUI 3.2.0: e.file has async read() method
                content = await e.file.read()
                filename = e.file.name
                
                if not content:
                    return
                
                # Save uploaded file
                upload_dir = Path('./uploads')
                upload_dir.mkdir(exist_ok=True)
                video_path = upload_dir / (filename or 'uploaded_video.mp4')
                
                with open(video_path, 'wb') as f:
                    f.write(content)
                
                # Show progress
                processing['active'] = True
                progress_bar.visible = True
                status_label.visible = True
                status_label.set_text('Analyzing video...')
                
                def update_progress(percent):
                    progress_bar.set_value(percent / 100)
                    status_label.set_text(f'Processing: {percent:.0f}%')
                
                # Process video with frequency analysis
                result = process_video_with_frequency(str(video_path), update_progress, ENABLE_PREPROCESSING)
                
                # Complete progress
                progress_bar.set_value(1.0)
                status_label.set_text('Complete!')
                
                # Show output area
                display_output(result, output_container, output_area)
                
                # Hide progress after delay
                async def hide_progress():
                    await asyncio.sleep(0.5)
                    progress_bar.visible = False
                    status_label.visible = False
                    processing['active'] = False
                
                await hide_progress()
                
            except Exception as e:
                print(f"Upload error: {e}")
                import traceback
                traceback.print_exc()
                processing['active'] = False
                progress_bar.visible = False
                status_label.visible = False
        
        def display_output(result, container, area):
            """Display enhanced results with frequency analysis"""
            area.visible = True
            container.clear()
            
            with container:
                # Error handling
                if 'error' in result:
                    ui.label(f'Error: {result["error"]}').classes('text-red-500 text-center')
                    return
                
                # Display preprocessing status
                if result.get('preprocessed', False) and ENABLE_PREPROCESSING and MEDIAPIPE_AVAILABLE:
                    ui.label('✅ Video was preprocessed (background removed)').classes('text-green-600 text-sm font-medium mb-2')
                
                # Display raw sequence
                ui.label('Raw Detection:').classes('text-gray-600 text-sm font-medium mt-2')
                ui.label(result.get('raw_sequence', '')).classes('font-mono text-gray-800 bg-gray-100 p-2 rounded w-full text-center')
                
                # Display frequency analysis
                if 'frequency_groups' in result:
                    groups_str = ', '.join([f"({letter},{count})" for letter, count in result['frequency_groups']])
                    ui.label('Frequency Analysis:').classes('text-gray-600 text-sm font-medium mt-4')
                    ui.label(groups_str).classes('font-mono text-blue-600 bg-blue-50 p-2 rounded w-full text-center')
                
                # Display AI interpretation
                interpretation_text = result.get('interpretation', 'No interpretation available')
                ui.label('AI Interpretation:').classes('text-gray-600 text-sm font-medium mt-4')
                ui.label(interpretation_text).classes('ai-output-text mt-2 text-center')
                
                # Display confidence
                if 'confidence' in result and result['confidence'] != 'N/A':
                    confidence_color = 'text-green-500' if result['confidence'] == 'HIGH' else 'text-yellow-500' if result['confidence'] == 'MEDIUM' else 'text-red-500'
                    ui.label(f'Confidence: {result["confidence"]}').classes(f'{confidence_color} text-sm mt-2 text-center')
                
                # Display reasoning if available
                if 'reasoning' in result and result['reasoning']:
                    ui.label('Analysis:').classes('text-gray-600 text-sm font-medium mt-4')
                    ui.label(result['reasoning']).classes('text-gray-600 text-sm italic text-center')
                
                # Display alternatives if available
                if 'alternatives' in result and result['alternatives']:
                    ui.label('Alternative interpretations:').classes('text-gray-600 text-sm font-medium mt-4')
                    for alt in result['alternatives'][:3]:  # Show top 3 alternatives
                        ui.label(f'• {alt}').classes('text-gray-600 text-center')

# ============================================================================
# Console Interface
# ============================================================================

def console_interface():
    """Command-line interface for the ASL analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Video Analyzer')
    parser.add_argument('--video', '-v', default='../content/video.mp4', help='Path to video file')
    parser.add_argument('--gap', '-g', type=int, default=10, help='Frame gap (process every Nth frame)')
    parser.add_argument('--model', '-m', default='../model/retrained_asl_model.pt', help='Path to model')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--outlier', '-o', type=float, default=1.5, help='Outlier std threshold')
    parser.add_argument('--api-key', '-k', help='Gemini API key')
    parser.add_argument('--gemini-model', '-gm', help='Gemini model name')
    parser.add_argument('--no-preprocess', action='store_true', help='Disable video preprocessing')
    
    args = parser.parse_args()
    
    print("🚀 Starting ASL Video Analyzer (Console Mode)")
    print("=" * 50)
    
    analyzer = ASLAnalyzer(
        video_path=args.video,
        frame_gap=args.gap,
        model_path=args.model,
        gemini_api_key=args.api_key,
        gemini_model=args.gemini_model,
        confidence_threshold=args.confidence,
        outlier_std_threshold=args.outlier,
        enable_preprocessing=not args.no_preprocess
    )
    
    result = analyzer.analyze_video()
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(f"\n✅ Analysis Complete!")
        if result.get('preprocessed'):
            print(f"✓ Video was preprocessed")
        print(f"Raw detection: {result['raw_sequence']}")
        print(f"Frequency analysis: {', '.join([f'({l},{c})' for l, c in result['frequency_groups']])}")
        print(f"AI Interpretation: {result['interpretation']}")
        print(f"Confidence: {result['confidence']}")
        if result.get('reasoning'):
            print(f"Reasoning: {result['reasoning']}")

# ============================================================================
# Application Startup
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run in console mode if arguments provided
        console_interface()
    else:
        # Run web interface
        print("🚀 Starting ASL Video Analyzer (Web Mode)...")
        
        # Initialize models
        init_models()
        
        # Create uploads directory
        Path('./uploads').mkdir(exist_ok=True)
        Path('./preprocessed').mkdir(exist_ok=True)
        
        # Check MediaPipe availability
        if MEDIAPIPE_AVAILABLE and ENABLE_PREPROCESSING:
            print("✅ Background removal preprocessing enabled")
        elif not MEDIAPIPE_AVAILABLE and ENABLE_PREPROCESSING:
            print("⚠ Background removal disabled (MediaPipe not installed)")
            print("   To enable: pip install mediapipe")
        
        print("\n🌟 Application ready! Open your browser to http://localhost:8080")
        print("   Use command-line arguments for console mode.")
        
        # Start NiceGUI
        ui.run(
            title='ASL Video Analyzer',
            port=8080,
            reload=False,
            show=False,
            favicon='🤟'
        )