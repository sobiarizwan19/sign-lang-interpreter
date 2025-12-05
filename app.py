import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import google.generativeai as genai
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import shutil

MODEL_PATH = "./model/sign-detection.pt"
GAP = 5
CONF_THRESHOLD = 0.5
GEMINI_API_KEY = "AIzaSyCv2XlAHLKQBCp6TzGk1GDiGLJ-EJ0mJ_g"
GEMINI_MODEL = "gemini-2.5-flash"
FILTER_THRESHOLD_PERCENT_50 = 50  # First filter: 50% less rigorous
FILTER_THRESHOLD_PERCENT_20 = 20  # Second filter: 20% 
NO_HAND_CONFIDENCE_THRESHOLD = 0.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLVideoDetector:
    def __init__(self, model_path, gap, conf_threshold):
        self.GAP = gap
        self.conf_threshold = conf_threshold
        self.no_hand_threshold = NO_HAND_CONFIDENCE_THRESHOLD
        
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
            return "SPACE", 0
        
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        max_confidence = 0
        best_class = None
        
        for class_id, confidence in zip(class_ids, confidences):
            if confidence > max_confidence:
                max_confidence = confidence
                best_class = self.class_names[class_id]
        
        # New confidence-based logic:
        # IF CONFIDENCE OF HIGHEST CONFIDENT BOX IS ABOVE 0.5 THEN ADD ALPHABETS
        if max_confidence > self.conf_threshold:
            return best_class, max_confidence
        elif max_confidence < self.no_hand_threshold:
            return "SPACE", max_confidence
        # OTHERWISE IGNORE
        else:
            return None, max_confidence
    
    def compress_consecutive_detections(self):
        if not self.detection_history:
            return []
        
        compressed = []
        current_letter = None
        current_count = 0
        
        for frame_num, letter, confidence in self.detection_history:
            if letter != current_letter:
                if current_letter is not None:
                    compressed.append((current_letter, current_count))
                current_letter = letter
                current_count = 1
            else:
                current_count += 1
        
        if current_letter is not None:
            compressed.append((current_letter, current_count))
        
        return compressed
    
    def filter_by_threshold(self, compressed_detections, threshold_percent):
        if not compressed_detections:
            return []
        
        non_space_detections = [det for det in compressed_detections if det[0] != 'SPACE']
        
        if not non_space_detections:
            return compressed_detections
        
        counts = [det[1] for det in non_space_detections]
        mean_count = sum(counts) / len(counts)
        threshold = mean_count * (1 - threshold_percent/100)
        logger.info(f"Filtering with threshold: {threshold:.2f} (Mean: {mean_count:.2f}, Percent: {threshold_percent}%)")
        filtered = []
        for det in compressed_detections:
            if det[1] >= threshold:
                filtered.append(det)
        
        return filtered
    
    def ask_gemini(self, filter_50_format, filter_20_format):
        if self.gemini_model is None:
            return "Gemini API key not configured"
        
        prompt = f"""
        I have ASL (American Sign Language) detection results in format (letter,count).
        Each (letter,count) pair represents a single letter held for 'count' consecutive frames.
        The sequence of pairs represents the order of letters in the ASL message.
        When no hand is detected, it's represented as 'SPACE' which indicates a pause between words.
        
        I'm providing you with TWO filtered versions of the same detection:
        1. FILTER 50%: Less rigorous filtering (50% threshold)
        2. FILTER 20%: More rigorous filtering (20% threshold)
        
        Make a valid English word, phrase or sentence from it.
        SENTANCE ,PHRASE OR WORD MUST BE IN ENGLISH LANGUAGE,VALID AND MAKE SENSE.
        Consider both filtered versions to make the best interpretation.
        
        FOLLOWING ALPHABETS ARE MORE LIKELY TO BE SWAPPED:
        'M' and 'N'
        'A' and 'Y'
        'O' and 'C'
        FOLLOWING ALPHABETS ARE MORE LIKELY TO BE MISSED:
        'D','Z','J'
        FILTER 50%: {filter_50_format}
        FILTER 20%: {filter_20_format}
        
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
        if video_path:
            self.setup_video(video_path)
        
        self.detection_history = []
        
        frame_count = 0
        processed_count = 0
        detection_count = 0
        space_count = 0
        ignored_count = 0
        
        logger.info(f"Starting video processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if (frame_count - 1) % self.GAP != 0:
                continue
            
            processed_count += 1
            
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
            
            # Only add to history if not ignored (None)
            if best_class is not None:
                self.detection_history.append((processed_count, best_class, confidence))
                
                if best_class == "SPACE":
                    space_count += 1
                    logger.info(f"Frame {frame_count}: No hand detected (conf: {confidence:.2f}) - adding SPACE")
                else:
                    detection_count += 1
                    logger.info(f"Frame {frame_count}: Detected '{best_class}' with confidence {confidence:.2f}")
            else:
                ignored_count += 1
                logger.info(f"Frame {frame_count}: Ignored detection (conf: {confidence:.2f})")
        
        self.cap.release()
        logger.info(f"Processing complete. Total frames: {frame_count}, Processed: {processed_count}")
        logger.info(f"Sign detections: {detection_count}, Space detections: {space_count}, Ignored: {ignored_count}")
        return self.get_ai_interpretation()
    
    def get_ai_interpretation(self):
        if not self.detection_history:
            logger.warning("No detections found in video.")
            return "No signs detected"
        
        # Get compressed version
        compressed = self.compress_consecutive_detections()
        
        # Apply both filters
        filter_50_result = self.filter_by_threshold(compressed, FILTER_THRESHOLD_PERCENT_50)
        filter_20_result = self.filter_by_threshold(compressed, FILTER_THRESHOLD_PERCENT_20)
        
        logger.info(f"Compressed {len(self.detection_history)} detections to {len(compressed)} segments")
        logger.info(f"Filter 50%: {len(filter_50_result)} segments")
        logger.info(f"Filter 20%: {len(filter_20_result)} segments")
        
        # Format both filtered results
        filter_50_format = " ".join([f"({letter},{count})" for letter, count in filter_50_result]) if filter_50_result else "No results"
        filter_20_format = " ".join([f"({letter},{count})" for letter, count in filter_20_result]) if filter_20_result else "No results"
        
        logger.info(f"Filter 50%: {filter_50_format}")
        logger.info(f"Filter 20%: {filter_20_format}")
        
        interpretation = self.ask_gemini(filter_50_format, filter_20_format)
        logger.info(f"Gemini interpretation: {interpretation}")
        
        return interpretation

detector = ASLVideoDetector(
    model_path=MODEL_PATH,
    gap=GAP,
    conf_threshold=CONF_THRESHOLD
)

app = FastAPI(title="ASL Translator API")

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        return JSONResponse(content={"error": "Invalid file type. Please upload a video file."}, status_code=400)
    
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        interpretation = detector.run_detection(temp_file_path)
        
        return JSONResponse(content={"interpretation": interpretation})
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 50)
    logger.info("STARTING ASL TRANSLATOR API")
    logger.info("=" * 50)
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Frame sampling: Every {GAP} frames")
    logger.info(f"Confidence threshold: {CONF_THRESHOLD}")
    logger.info(f"No hand threshold: {NO_HAND_CONFIDENCE_THRESHOLD}")
    logger.info(f"Filter 1 (50%): Less rigorous filtering")
    logger.info(f"Filter 2 (20%): More rigorous filtering")
    logger.info("SENDING BOTH FILTERED VERSIONS TO LLM")
    logger.info("=" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8002)