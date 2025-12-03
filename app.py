#!/usr/bin/env python3
"""
ASL Video Analyzer with Dual Priority Frequency Analysis
Complete project in a single file
"""

import cv2
import os
import tempfile
import sys
import math
from pathlib import Path
import asyncio
from collections import defaultdict, Counter
from nicegui import ui, app
from dotenv import load_dotenv
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image

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
CONFIDENCE_THRESHOLD_2 = float(os.getenv('CONFIDENCE_THRESHOLD_2', '0.0'))  # NO threshold for secondary
OUTLIER_STD_THRESHOLD = float(os.getenv('OUTLIER_STD_THRESHOLD', '1.5'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

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
            'top2_class': class_names[top5_indices[1]] if len(top5_indices) > 1 else None,
            'top2_confidence': top5_conf[1] if len(top5_conf) > 1 else None,
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
# Sentence Predictor (Gemini LLM) with Dual Priority Support
# ============================================================================

class SentencePredictor:
    """
    Predicts sentences from ASL alphabet sequences using Google Gemini
    with support for dual priority frequency analysis
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
        
        # Safety settings to prevent blocking - MORE PERMISSIVE
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
                # Initialize with safety settings
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=safety_settings
                )
                # Test with simple prompt
                test_response = self.model.generate_content(
                    "Hello, this is a test.",
                    safety_settings=safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=100,
                    )
                )
                # Verify we can access text
                _ = test_response.text
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
            print("⚠ Warning: Could not initialize Gemini model")
    
    def predict_dual_priority(self, primary_sequence: str, secondary_sequence: str) -> dict:
        """
        Predict sentence from dual priority sequences
        """
        if not primary_sequence or primary_sequence.strip() == "":
            return self._create_fallback_result(primary_sequence, secondary_sequence, "Empty sequence")
        
        if self.model is None:
            return self._create_fallback_result(primary_sequence, secondary_sequence, "Model not initialized")
        
        prompt = self._create_safe_dual_priority_prompt(primary_sequence, secondary_sequence)
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Very low temperature for consistent results
                    max_output_tokens=200,
                    top_p=0.8,
                    top_k=40
                )
            )
            
            # Get response text safely
            try:
                response_text = response.text
            except Exception as e:
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    block_reason = str(response.prompt_feedback)
                    return self._create_fallback_result(primary_sequence, secondary_sequence, f"Blocked: {block_reason}")
                else:
                    return self._create_fallback_result(primary_sequence, secondary_sequence, f"No response: {str(e)}")
            
            return self._parse_dual_response_simple(response_text, primary_sequence, secondary_sequence)
            
        except Exception as e:
            error_msg = str(e)
            print(f"API Error: {error_msg[:100]}")
            return self._create_fallback_result(primary_sequence, secondary_sequence, f"API error: {error_msg[:50]}")
    
    def _create_safe_dual_priority_prompt(self, primary_sequence: str, secondary_sequence: str) -> str:
        """Create SAFE prompt for dual priority interpretation"""
        
        # Parse primary sequence
        primary_pairs = []
        for item in primary_sequence.split():
            if ':' in item:
                letter, count = item.split(':')
                primary_pairs.append(f"{letter}({count})")
            else:
                primary_pairs.append(f"{letter}(1)")
        
        primary_str = " ".join(primary_pairs)
        
        # Parse secondary sequence (if exists)
        secondary_str = ""
        if secondary_sequence and secondary_sequence.strip():
            secondary_pairs = []
            for item in secondary_sequence.split():
                if ':' in item:
                    letter, count = item.split(':')
                    secondary_pairs.append(f"{letter}({count})")
                else:
                    secondary_pairs.append(f"{letter}(1)")
            secondary_str = " ".join(secondary_pairs)
        
        prompt = f"""Interpret these ASL letter sequences:

Primary sequence (more confident): {primary_str}
Secondary sequence (alternative possibilities): {secondary_str if secondary_str else "None"}

Interpret these ASL finger-spelled letters into the most likely word or short phrase.
The secondary sequence shows possible alternatives for each position.
Use both sequences to determine the most likely interpretation.

Output only the interpreted word/phrase.

Examples:
- Primary: A(5) B(3) C(2), Secondary: B(2) C(3) D(1) -> "ABC"
- Primary: H(4) E(3) L(2) L(1) O(4), Secondary: H(4) S(1) L(2) L(1) O(4) -> "HELLO"
- Primary: T(3) H(2) A(4) N(3) K(2), Secondary: T(3) N(2) A(4) M(1) K(2) -> "THANK"

Your interpretation:"""
        return prompt
    
    def _parse_dual_response_simple(self, response_text: str, primary_sequence: str, secondary_sequence: str) -> dict:
        """Parse response into simple format"""
        # Clean the response
        response_text = response_text.strip().strip('"\'').split('\n')[0].strip()
        
        # Calculate confidence based on sequence quality
        confidence = self._calculate_confidence(primary_sequence, secondary_sequence)
        
        # Generate simple interpretation from primary sequence
        simple_primary = self._sequence_to_simple_text(primary_sequence)
        simple_secondary = self._sequence_to_simple_text(secondary_sequence) if secondary_sequence else ""
        
        return {
            'interpretation': response_text if response_text else simple_primary,
            'alternatives': [simple_primary, simple_secondary] if simple_secondary else [simple_primary],
            'confidence': confidence,
            'reasoning': f"Interpreted from ASL sequences. Primary: {simple_primary}" + (f", Secondary: {simple_secondary}" if simple_secondary else ""),
            'primary_sequence': primary_sequence,
            'secondary_sequence': secondary_sequence,
            'simple_primary': simple_primary,
            'simple_secondary': simple_secondary
        }
    
    def _calculate_confidence(self, primary_sequence: str, secondary_sequence: str) -> str:
        """Calculate confidence based on sequences"""
        # Count weights in primary sequence
        primary_weight = 0
        for item in primary_sequence.split():
            if ':' in item:
                _, count = item.split(':')
                primary_weight += int(count)
        
        # High confidence if primary has good weight
        if primary_weight > 20:
            return "HIGH"
        elif primary_weight > 10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _create_fallback_result(self, primary_sequence: str, secondary_sequence: str, reason: str) -> dict:
        """Create fallback result when LLM fails"""
        simple_primary = self._sequence_to_simple_text(primary_sequence)
        simple_secondary = self._sequence_to_simple_text(secondary_sequence) if secondary_sequence else ""
        
        # Try to form a simple word from primary sequence
        interpretation = self._form_simple_word(simple_primary)
        
        return {
            'interpretation': interpretation,
            'alternatives': [simple_primary, simple_secondary] if simple_secondary else [simple_primary],
            'confidence': 'MEDIUM',
            'reasoning': f"Using simple analysis: {reason}",
            'primary_sequence': primary_sequence,
            'secondary_sequence': secondary_sequence,
            'simple_primary': simple_primary,
            'simple_secondary': simple_secondary
        }
    
    def _sequence_to_simple_text(self, frequency_sequence: str) -> str:
        """Convert frequency sequence to simple text (single letters)"""
        letters = []
        for item in frequency_sequence.split():
            if ':' in item:
                letter, _ = item.split(':')
                letters.append(letter)
            else:
                letters.append(item)
        return ''.join(letters)
    
    def _form_simple_word(self, letters: str) -> str:
        """Try to form a simple word from letters"""
        # Common ASL words to check
        common_words = [
            "HELLO", "HI", "YES", "NO", "THANK", "YOU", "PLEASE", "SORRY",
            "NAME", "MY", "YOUR", "WHAT", "HOW", "WHERE", "WHEN", "WHY",
            "GOOD", "BAD", "HAPPY", "SAD", "LOVE", "LIKE", "NEED", "WANT",
            "HELP", "STOP", "GO", "COME", "EAT", "DRINK", "SLEEP", "WORK",
            "HOME", "SCHOOL", "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER"
        ]
        
        letters_upper = letters.upper()
        
        # Check if sequence matches start of any common word
        for word in common_words:
            if word.startswith(letters_upper) or letters_upper in word:
                return word
        
        # Return the letters as-is
        return letters_upper if letters_upper else "No interpretation"

# ============================================================================
# Main ASL Analyzer with Dual Priority - ALWAYS INCLUDE SECONDARY
# ============================================================================

class ASLAnalyzer:
    def __init__(self, 
                 video_path="../content/video.mp4", 
                 frame_gap=10,
                 model_path="../model/retrained_asl_model.pt",
                 gemini_api_key=None,
                 gemini_model=None,
                 confidence_threshold=0.5,
                 confidence_threshold_2=0.0,  # NO THRESHOLD FOR SECONDARY
                 outlier_std_threshold=1.5):
        """
        Initialize the ASL analyzer with dual priority processing
        ALWAYS include secondary prediction if primary is detected
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
        self.confidence_threshold_2 = confidence_threshold_2  # 0.0 = NO THRESHOLD
        self.outlier_std_threshold = outlier_std_threshold
        
        self.classifier = None
        self.sentence_predictor = None
        self.primary_predictions = []
        self.secondary_predictions = []
        self.primary_frequency = []
        self.secondary_frequency = []
        
        # Initialize components
        self._init_classifier()
        self._init_llm()
        
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
            if self.sentence_predictor.model:
                print(f"✓ Gemini predictor initialized: {self.sentence_predictor.model_name}")
            else:
                print("⚠ Warning: LLM not available, using simple analysis")
        except Exception as e:
            self.sentence_predictor = None
            print(f"⚠ Warning: Could not initialize LLM: {e}")
    
    def analyze_video(self, progress_callback=None):
        """
        Analyze the entire video and extract sign language sequence
        """
        # Check if video file exists
        if not os.path.exists(self.video_path):
            print(f"Video file not found: {self.video_path}")
            return {
                'error': f'Video file not found: {self.video_path}',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Cannot open video file: {self.video_path}")
            return {
                'error': f'Cannot open video: {self.video_path}',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames total")
        print(f"Processing every {self.frame_gap} frames...")
        print(f"Primary confidence threshold: {self.confidence_threshold}")
        print(f"Secondary confidence threshold: {self.confidence_threshold_2} (NO THRESHOLD - ALWAYS INCLUDE)")
        
        # Track predictions
        primary_list = []
        secondary_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every frame_gap frames
            if frame_count % self.frame_gap == 0:
                # Get dual predictions for this frame
                predictions = self._predict_frame_dual(frame)
                
                if predictions['primary']:
                    primary_list.append(predictions['primary'])
                    # ALWAYS include secondary if we have primary
                    if predictions['secondary']:
                        secondary_list.append(predictions['secondary'])
                    else:
                        secondary_list.append('?')  # Placeholder if no secondary
                else:
                    primary_list.append(None)
                    secondary_list.append(None)
            
            frame_count += 1
            
            # Progress callback
            if progress_callback and frame_count % (self.frame_gap * 10) == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
        
        cap.release()
        
        # Store raw predictions
        self.primary_predictions = [p for p in primary_list if p is not None]
        self.secondary_predictions = [s for s in secondary_list if s is not None]
        
        print(f"\nPrimary predictions: {''.join(self.primary_predictions)}")
        print(f"Secondary predictions: {''.join(self.secondary_predictions)}")
        print(f"Secondary includes ? for missing predictions")
        
        if not self.primary_predictions:
            print("No primary signs detected in the video")
            return {
                'error': 'No signs detected in video',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        # Process frequency analysis
        self._process_dual_frequency_analysis()
        
        # Get interpretation
        return self._get_dual_interpretation()
    
    def _process_dual_frequency_analysis(self):
        """Process dual predictions with frequency analysis"""
        print("\nProcessing frequency analysis...")
        
        # Process primary sequence
        if self.primary_predictions:
            # Group consecutive identical letters
            primary_groups = []
            if self.primary_predictions:
                current_letter = self.primary_predictions[0]
                current_count = 1
                
                for i in range(1, len(self.primary_predictions)):
                    if self.primary_predictions[i] == current_letter:
                        current_count += 1
                    else:
                        primary_groups.append((current_letter, current_count))
                        current_letter = self.primary_predictions[i]
                        current_count = 1
                
                primary_groups.append((current_letter, current_count))
            
            # Merge consecutive same letters
            merged_primary = []
            if primary_groups:
                current_letter, current_total = primary_groups[0]
                
                for i in range(1, len(primary_groups)):
                    letter, count = primary_groups[i]
                    if letter == current_letter:
                        current_total += count
                    else:
                        merged_primary.append((current_letter, current_total))
                        current_letter = letter
                        current_total = count
                
                merged_primary.append((current_letter, current_total))
            
            self.primary_frequency = merged_primary
            print(f"Primary frequency: {self.primary_frequency}")
        
        # Process secondary sequence - INCLUDES '?' PLACEHOLDERS
        if self.secondary_predictions:
            # Group consecutive identical letters
            secondary_groups = []
            current_letter = self.secondary_predictions[0]
            current_count = 1
            
            for i in range(1, len(self.secondary_predictions)):
                if self.secondary_predictions[i] == current_letter:
                    current_count += 1
                else:
                    secondary_groups.append((current_letter, current_count))
                    current_letter = self.secondary_predictions[i]
                    current_count = 1
            
            secondary_groups.append((current_letter, current_count))
            
            # Merge consecutive same letters
            merged_secondary = []
            if secondary_groups:
                current_letter, current_total = secondary_groups[0]
                
                for i in range(1, len(secondary_groups)):
                    letter, count = secondary_groups[i]
                    if letter == current_letter:
                        current_total += count
                    else:
                        merged_secondary.append((current_letter, current_total))
                        current_letter = letter
                        current_total = count
                
                merged_secondary.append((current_letter, current_total))
            
            self.secondary_frequency = merged_secondary
            print(f"Secondary frequency: {self.secondary_frequency}")
    
    def _predict_frame_dual(self, frame):
        """
        Predict sign for a frame - ALWAYS include secondary if primary exists
        NO THRESHOLD for secondary prediction
        """
        if self.classifier is None:
            return {'primary': None, 'secondary': None}
        
        try:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            # Get prediction
            result = self.classifier.predict_single_image(temp_path, show=False, save=False)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Extract predictions
            primary_letter = result['top_class']
            primary_confidence = float(result['top_confidence'])
            
            secondary_letter = result['top2_class']
            secondary_confidence = float(result['top2_confidence']) if result['top2_confidence'] else 0.0
            
            # Define valid ASL alphabet letters
            valid_asl_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            primary_pred = None
            secondary_pred = None
            
            # Check primary prediction
            if (len(primary_letter) == 1 and 
                primary_letter.upper() in valid_asl_letters and 
                primary_confidence > self.confidence_threshold):
                
                primary_pred = primary_letter.upper()
                
                # ALWAYS include secondary if we have primary
                if secondary_letter and len(secondary_letter) == 1:
                    if secondary_letter.upper() in valid_asl_letters:
                        secondary_pred = secondary_letter.upper()
                    else:
                        secondary_pred = '?'  # Invalid letter
                else:
                    secondary_pred = '?'  # No secondary prediction
            
            return {
                'primary': primary_pred,
                'secondary': secondary_pred,
                'primary_confidence': primary_confidence,
                'secondary_confidence': secondary_confidence
            }
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'primary': None, 'secondary': None}
    
    def _get_dual_interpretation(self):
        """
        Get AI interpretation using DUAL priority sequences
        """
        if not self.primary_frequency:
            return {
                'error': 'No primary frequency sequence generated',
                'sequence': '',
                'interpretation': '',
                'confidence': 'LOW'
            }
        
        # Format sequences
        primary_str = " ".join([f"{letter}:{count}" for letter, count in self.primary_frequency])
        secondary_str = " ".join([f"{letter}:{count}" for letter, count in self.secondary_frequency]) if self.secondary_frequency else ""
        
        # Simple sequences for display
        simple_primary = "".join([letter for letter, _ in self.primary_frequency])
        simple_secondary = "".join([letter for letter, _ in self.secondary_frequency]) if self.secondary_frequency else ""
        
        print(f"\n=== FINAL SEQUENCES ===")
        print(f"Primary: {primary_str}")
        print(f"Simple primary: {simple_primary}")
        if secondary_str:
            print(f"Secondary: {secondary_str}")
            print(f"Simple secondary: {simple_secondary}")
        
        # Try to interpret with AI if available
        if self.sentence_predictor and self.sentence_predictor.model:
            try:
                result = self.sentence_predictor.predict_dual_priority(primary_str, secondary_str)
                
                return {
                    'raw_primary': "".join(self.primary_predictions),
                    'raw_secondary': "".join(self.secondary_predictions),
                    'frequency_primary': primary_str,
                    'frequency_secondary': secondary_str,
                    'simple_primary': simple_primary,
                    'simple_secondary': simple_secondary,
                    'interpretation': result['interpretation'],
                    'confidence': result['confidence'],
                    'reasoning': result.get('reasoning', ''),
                    'alternatives': result.get('alternatives', []),
                    'primary_groups': self.primary_frequency,
                    'secondary_groups': self.secondary_frequency,
                    'analysis_type': 'DUAL_PRIORITY'
                }
                
            except Exception as e:
                print(f"Error in AI interpretation: {e}")
                # Fallback to simple interpretation
                return self._get_simple_interpretation(primary_str, secondary_str, simple_primary, simple_secondary)
        else:
            # Use simple analysis
            return self._get_simple_interpretation(primary_str, secondary_str, simple_primary, simple_secondary)
    
    def _get_simple_interpretation(self, primary_str, secondary_str, simple_primary, simple_secondary):
        """Get simple interpretation without LLM"""
        # Try to form a word from the primary sequence
        interpretation = simple_primary
        
        # Check if it looks like a common word
        common_prefixes = ["HEL", "THA", "YOU", "PLE", "SOR", "NAM", "GOO", "BAD", "LOV", "LIK"]
        
        for prefix in common_prefixes:
            if simple_primary.startswith(prefix):
                # Try to complete the word
                if prefix == "HEL":
                    interpretation = "HELLO"
                elif prefix == "THA":
                    interpretation = "THANK"
                elif prefix == "YOU":
                    interpretation = "YOU"
                elif prefix == "PLE":
                    interpretation = "PLEASE"
                elif prefix == "SOR":
                    interpretation = "SORRY"
                elif prefix == "NAM":
                    interpretation = "NAME"
                elif prefix == "GOO":
                    interpretation = "GOOD"
                elif prefix == "BAD":
                    interpretation = "BAD"
                elif prefix == "LOV":
                    interpretation = "LOVE"
                elif prefix == "LIK":
                    interpretation = "LIKE"
                break
        
        return {
            'raw_primary': "".join(self.primary_predictions),
            'raw_secondary': "".join(self.secondary_predictions),
            'frequency_primary': primary_str,
            'frequency_secondary': secondary_str,
            'simple_primary': simple_primary,
            'simple_secondary': simple_secondary,
            'interpretation': interpretation,
            'confidence': 'MEDIUM',
            'reasoning': f'Simple analysis: Primary={simple_primary}, Secondary={simple_secondary}',
            'alternatives': [simple_secondary] if simple_secondary else [],
            'primary_groups': self.primary_frequency,
            'secondary_groups': self.secondary_frequency,
            'analysis_type': 'SIMPLE_ANALYSIS'
        }

# ============================================================================
# Web Application
# ============================================================================

# Global objects
classifier = None
sentence_predictor = None

def init_models():
    """Initialize the ASL classifier and sentence predictor"""
    global classifier, sentence_predictor
    
    try:
        # Initialize classifier
        classifier = ASLClassifier(model_path=MODEL_PATH)
        print(f"✓ Classifier loaded: {classifier.model_path}")
        
        # Initialize sentence predictor with Gemini
        if GEMINI_API_KEY:
            sentence_predictor = SentencePredictor(
                api_key=GEMINI_API_KEY,
                model_name=GEMINI_MODEL
            )
            if sentence_predictor.model:
                print(f"✓ Gemini predictor initialized: {sentence_predictor.model_name}")
            else:
                print("⚠ Warning: Gemini model not available")
        else:
            print("⚠ Warning: GEMINI_API_KEY not found")
            sentence_predictor = None
            
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        raise

def process_video_with_dual_priority(video_path: str, progress_callback=None):
    """
    Process video with DUAL PRIORITY frequency analysis
    ALWAYS include secondary predictions
    """
    analyzer = ASLAnalyzer(
        video_path=video_path,
        frame_gap=FRAME_GAP,
        model_path=MODEL_PATH,
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        confidence_threshold_2=CONFIDENCE_THRESHOLD_2,  # 0.0 = NO THRESHOLD
        outlier_std_threshold=OUTLIER_STD_THRESHOLD
    )
    
    return analyzer.analyze_video(progress_callback)

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
    width: 100%;
    max-width: 800px;
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
    font-size: 2.5rem;
    font-weight: 600;
    line-height: 1.3;
    color: #1f2937;
    text-align: center;
    margin: 20px 0;
    padding: 20px;
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-radius: 12px;
    border: 2px solid #3b82f6;
}

.sequence-display {
    font-family: 'Courier New', monospace;
    font-size: 1.2rem;
    background: #f8fafc;
    padding: 10px 15px;
    border-radius: 8px;
    margin: 10px 0;
    border: 1px solid #e2e8f0;
}

.analysis-section {
    margin: 20px 0;
    padding: 15px;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 4px solid #3b82f6;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #475569;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.confidence-high { 
    color: #10b981; 
    font-weight: bold;
    background: #d1fae5;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
}
.confidence-medium { 
    color: #f59e0b; 
    font-weight: bold;
    background: #fef3c7;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
}
.confidence-low { 
    color: #ef4444; 
    font-weight: bold;
    background: #fee2e2;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
}

.reasoning-box {
    background: #f1f5f9;
    padding: 15px;
    border-radius: 8px;
    font-size: 0.95rem;
    color: #475569;
    font-style: italic;
    margin: 15px 0;
    border-left: 3px solid #94a3b8;
}

.alternative-item {
    background: #e0f2fe;
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 6px;
    font-size: 0.9rem;
    color: #0369a1;
    border-left: 3px solid #0ea5e9;
}

/* Primary/Secondary indicators */
.primary-label {
    background: #3b82f6;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.secondary-label {
    background: #8b5cf6;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.secondary-placeholder {
    color: #94a3b8;
    font-style: italic;
}

.hidden {
    display: none;
}
"""

@ui.page('/')
async def main_page():
    """Main page with upload and output"""
    
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
            """Handle video upload and processing"""
            if processing['active']:
                return
            
            try:
                # Get file
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
                
                # Process video
                result = process_video_with_dual_priority(str(video_path), update_progress)
                
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
            """Display results"""
            area.visible = True
            container.clear()
            
            with container:
                # Error handling
                if 'error' in result:
                    ui.label(f'❌ Error: {result["error"]}').classes('text-red-500 text-center text-lg')
                    return
                
                # Show analysis type
                analysis_type = result.get('analysis_type', '')
                if analysis_type == 'DUAL_PRIORITY':
                    ui.label('🔬 Dual Priority Analysis').classes('text-blue-600 font-bold text-xl text-center mb-6')
                elif analysis_type == 'SIMPLE_ANALYSIS':
                    ui.label('📊 Simple Analysis').classes('text-purple-600 font-bold text-xl text-center mb-6')
                
                # Main AI Interpretation
                interpretation = result.get('interpretation', 'No interpretation')
                ui.label(interpretation).classes('ai-output-text')
                
                # Confidence indicator
                confidence = result.get('confidence', 'MEDIUM')
                confidence_class = {
                    'HIGH': 'confidence-high',
                    'MEDIUM': 'confidence-medium', 
                    'LOW': 'confidence-low'
                }.get(confidence, 'confidence-medium')
                
                with ui.row().classes('items-center justify-center gap-2 mb-6'):
                    ui.label('Confidence:').classes('text-gray-600')
                    ui.label(confidence).classes(confidence_class)
                
                # Primary Sequence Section
                with ui.column().classes('analysis-section w-full'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Primary Detection').classes('primary-label')
                        ui.label('(High Confidence)').classes('text-gray-600 text-sm')
                    
                    # Show frequency analysis
                    if 'primary_groups' in result:
                        groups_str = ' → '.join([f"{letter}({count})" for letter, count in result['primary_groups']])
                        ui.label(groups_str).classes('sequence-display')
                    
                    # Show simple primary
                    simple_primary = result.get('simple_primary', '')
                    if simple_primary:
                        ui.label(f'Simple: {simple_primary}').classes('text-gray-700 text-sm mt-1')
                
                # Secondary Sequence Section (ALWAYS SHOW if we have primary)
                with ui.column().classes('analysis-section w-full'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('Secondary Detection').classes('secondary-label')
                        ui.label('(Always Included)').classes('text-gray-600 text-sm')
                    
                    # Show frequency analysis
                    if 'secondary_groups' in result:
                        groups_str = ' → '.join([f"{letter}({count})" for letter, count in result['secondary_groups']])
                        # Highlight '?' placeholders
                        if '?' in groups_str:
                            ui.label(groups_str).classes('sequence-display secondary-placeholder')
                        else:
                            ui.label(groups_str).classes('sequence-display')
                    
                    # Show simple secondary
                    simple_secondary = result.get('simple_secondary', '')
                    if simple_secondary:
                        if '?' in simple_secondary:
                            ui.label(f'Simple: {simple_secondary}').classes('text-gray-500 text-sm mt-1 secondary-placeholder')
                        else:
                            ui.label(f'Simple: {simple_secondary}').classes('text-gray-700 text-sm mt-1')
                    
                    # Explanation
                    ui.label('Secondary predictions always included when primary is detected').classes('text-gray-500 text-xs mt-2 italic')
                
                # Reasoning
                if 'reasoning' in result and result['reasoning']:
                    with ui.column().classes('w-full mt-4'):
                        ui.label('Analysis:').classes('section-title')
                        ui.label(result['reasoning']).classes('reasoning-box')
                
                # Alternatives
                if 'alternatives' in result and result['alternatives']:
                    with ui.column().classes('w-full mt-4'):
                        ui.label('Alternative Interpretations:').classes('section-title')
                        for alt in result['alternatives']:
                            if alt and alt.strip():  # Only show non-empty alternatives
                                ui.label(alt).classes('alternative-item')

# ============================================================================
# Application Startup
# ============================================================================

if __name__ == '__main__':
    print("🚀 Starting ASL Video Analyzer...")
    print("⚠ Secondary predictions: ALWAYS INCLUDED when primary detected (NO threshold)")
    
    # Initialize models
    init_models()
    
    # Create uploads directory
    Path('./uploads').mkdir(exist_ok=True)
    
    print("🌟 Application ready! Open your browser to http://localhost:8080")
    
    # Start NiceGUI
    ui.run(
        title='ASL Video Analyzer',
        port=8080,
        reload=False,
        show=False,
        favicon='🤟'
    )