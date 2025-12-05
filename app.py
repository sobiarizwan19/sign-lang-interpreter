import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import google.generativeai as genai
import gradio as gr
import logging

# ===== CONFIG VARIABLES - CHANGE THESE =====
MODEL_PATH = "./model/sign-detection.pt"
GAP = 5
CONF_THRESHOLD = 0.5
GEMINI_API_KEY = "AIzaSyCv2XlAHLKQBCp6TzGk1GDiGLJ-EJ0mJ_g"
GEMINI_MODEL = "gemini-2.5-flash"
FILTER_THRESHOLD_PERCENT = 30  # Mean - 30% threshold
# ===========================================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLVideoDetector:
    def __init__(self, model_path, gap, conf_threshold):
        self.GAP = gap
        self.conf_threshold = conf_threshold
        
        # Gemini API setup
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("Gemini API key not set. Interpretation will be skipped.")
            self.gemini_model = None
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        logger.info(f"Model loaded: {self.model.names}")
        
        self.cap = None
        self.video_path = None
        self.class_names = self.model.names
        self.detection_history = []
        
    def setup_video(self, video_path):
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return False
        
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        
        if not self.cap.isOpened():
            logger.error("Could not open video file!")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video loaded: {video_path}")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(f"Config: GAP={self.GAP}, CONF={self.conf_threshold}")
        return True
    
    def get_top_detection(self, results):
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return None, 0
        
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        max_confidence = 0
        best_class = None
        
        for class_id, confidence in zip(class_ids, confidences):
            if confidence > self.conf_threshold and confidence > max_confidence:
                max_confidence = confidence
                best_class = self.class_names[class_id]
        
        return best_class, max_confidence
    
    def compress_consecutive_detections(self):
        if not self.detection_history:
            return []
        
        compressed = []
        current_letter = None
        current_count = 0
        start_frame = 0
        
        for frame_num, letter, _ in self.detection_history:
            if letter != current_letter:
                if current_letter is not None:
                    compressed.append((current_letter, current_count, start_frame, frame_num-1))
                current_letter = letter
                current_count = 1
                start_frame = frame_num
            else:
                current_count += 1
        
        if current_letter is not None:
            compressed.append((current_letter, current_count, start_frame, self.detection_history[-1][0]))
        
        return compressed
    
    def filter_by_threshold(self, compressed_detections):
        if not compressed_detections:
            return []
        
        counts = [det[1] for det in compressed_detections]
        mean_count = sum(counts) / len(counts)
        threshold = mean_count * (1 - FILTER_THRESHOLD_PERCENT/100)  # Mean - 30%
        
        filtered = [det for det in compressed_detections if det[1] >= threshold]
        
        return filtered, mean_count, threshold
    
    def ask_gemini(self, compressed_format):
        if self.gemini_model is None:
            return "Gemini API key not configured"
        
        prompt = f"""
        I have ASL (American Sign Language) detection results in format (letter,count).
        The letters should be in same order and counts represent consecutive detections.
        There might be outliers. Make a valid English word, phrase or sentence from it, which is grammatically correct and it makes full sense.
        
        Format: {compressed_format}
        
        Just return the interpretation without any explanations.
        """
        
        try:
            logger.info("Asking Gemini for interpretation...")
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"Gemini response received: {response.text[:50]}...")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error: {e}"
    
    def run_detection(self, video_path=None):
        """Run detection on video and return results"""
        if video_path:
            self.setup_video(video_path)
        
        # Reset history
        self.detection_history = []
        
        frame_count = 0
        processed_count = 0
        detection_count = 0
        
        logger.info(f"Starting video processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if (frame_count - 1) % self.GAP != 0:
                continue
            
            processed_count += 1
            
            # Log progress every 50 processed frames
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{self.total_frames//self.GAP} frames...")
            
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=0.45,
                imgsz=640,
                verbose=False,
                max_det=20
            )
            
            best_class, confidence = self.get_top_detection(results)
            if best_class is not None and confidence > self.conf_threshold:
                self.detection_history.append((processed_count, best_class, confidence))
                detection_count += 1
            
                logger.info(f"Frame {frame_count}: Detected '{best_class}' with confidence {confidence:.2f}")
        
        self.cap.release()
        logger.info(f"Processing complete. Total frames: {frame_count}, Processed: {processed_count}, Detections: {detection_count}")
        return self.get_results_summary()
    
    def get_results_summary(self):
        """Get formatted results summary - returns only raw and AI interpretation"""
        if not self.detection_history:
            logger.warning("No detections found in video.")
            return "No signs detected", "Please try another video"
        
        compressed = self.compress_consecutive_detections()
        filtered, _, _ = self.filter_by_threshold(compressed)
        
        logger.info(f"Compressed {len(self.detection_history)} detections to {len(compressed)} segments")
        logger.info(f"After filtering: {len(filtered)} segments remain")
        
        # Get raw letters
        raw_letters = "".join([letter for letter, _, _, _ in filtered])
        logger.info(f"Raw letters detected: {raw_letters}")
        
        # Get Gemini interpretation
        interpretation = ""
        if self.gemini_model is not None and filtered:
            compressed_format = " ".join([f"({letter},{count})" for letter, count, _, _ in filtered])
            logger.info(f"Sending to Gemini: {compressed_format}")
            interpretation = self.ask_gemini(compressed_format)
            logger.info(f"Gemini interpretation: {interpretation}")
        
        return raw_letters, interpretation

# Initialize detector
detector = ASLVideoDetector(
    model_path=MODEL_PATH,
    gap=GAP,
    conf_threshold=CONF_THRESHOLD
)

def process_video(video_file):
    """Process uploaded video file - with proper logging"""
    if video_file is None:
        logger.warning("No video file uploaded")
        return "Upload a video first", "Upload a video first"
    
    logger.info(f"Received video file for processing")
    
    try:
        # Get the file path
        video_path = video_file if isinstance(video_file, str) else video_file.name
        logger.info(f"Processing video: {video_path}")
        
        # Run detection
        logger.info("Starting detection process...")
        raw_letters, interpretation = detector.run_detection(video_path)
        logger.info("Detection process completed successfully")
        
        return raw_letters, interpretation
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", f"Error: {str(e)}"

# Create MINIMAL Gradio interface
with gr.Blocks(title="ASL Translator") as demo:
    # Heading only
    gr.Markdown("# ASL Translator")
    
    # Upload section
    video_input = gr.Video(label="Upload ASL Video")
    
    # Process button
    process_btn = gr.Button("Translate Video", variant="primary", size="lg")
    
    # Results section - only two outputs
    with gr.Row():
        with gr.Column():
            raw_output = gr.Textbox(
                label="Raw Detection", 
                placeholder="Detected letters will appear here...",
                interactive=False
            )
        
        with gr.Column():
            ai_output = gr.Textbox(
                label="AI Interpretation", 
                placeholder="AI interpretation will appear here...",
                interactive=False
            )
    
    # Connect button with progress indicator
    process_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[raw_output, ai_output],
        show_progress=True  # Show progress in UI
    )

# Launch the app with logging
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("STARTING ASL TRANSLATOR APPLICATION")
    logger.info("=" * 50)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Frame sampling: Every {GAP} frames")
    logger.info(f"Confidence threshold: {CONF_THRESHOLD}")
    logger.info(f"Filter threshold: {FILTER_THRESHOLD_PERCENT}% below mean")
    logger.info("=" * 50)
    
    try:
        demo.launch(
            share=False,
            show_error=True,
            quiet=False,  # Show Gradio's own logs too
            server_name="127.0.0.1",
            server_port=7860
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise