import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import google.generativeai as genai
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
GAP = int(os.getenv("GAP"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
NO_HAND_CONFIDENCE_THRESHOLD = float(os.getenv("NO_HAND_CONFIDENCE_THRESHOLD"))
PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD = float(
    os.getenv("PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD")
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ASLVideoDetector:
    def __init__(self, model_path, gap, conf_threshold):
        self.GAP = gap
        self.conf_threshold = conf_threshold
        self.no_hand_threshold = NO_HAND_CONFIDENCE_THRESHOLD
        self.percent_for_final_threshold = PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD
        
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        
        self.cap = None
        self.video_path = None
        self.class_names = self.model.names
        self.detection_history = []
        
        self.current_letter = None
        self.current_letter_start_frame = None
        self.current_letter_count = 0
        self.current_letter_confidences = []
        
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
        
        if max_confidence > self.conf_threshold:
            return best_class, max_confidence
        elif max_confidence < self.no_hand_threshold:
            return "SPACE", max_confidence
        else:
            return None, max_confidence
    
    def log_consecutive_detection(self, frame_num, letter, confidence):
        LOG_FMT = "Frame {start:>5}-{end:<5} | Detected {letter:<7} | Count:{count:>3}"

        if letter != self.current_letter:
            if self.current_letter is not None:
                end_frame = (
                    frame_num if self.current_letter_start_frame == frame_num - self.GAP
                    else frame_num - self.GAP
                )

                logger.info(LOG_FMT.format(
                    start=self.current_letter_start_frame,
                    end=end_frame,
                    letter=self.current_letter,
                    count=self.current_letter_count
                ))

            self.current_letter = letter
            self.current_letter_start_frame = frame_num
            self.current_letter_count = 1
            self.current_letter_confidences = [confidence]

        else:
            self.current_letter_count += 1
            self.current_letter_confidences.append(confidence)  
    
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
    
    def merge_consecutive_same(self, detections):
        if not detections:
            return []
        
        merged = []
        current_item = None
        current_total_count = 0
        
        for item, count in detections:
            if item != current_item:
                if current_item is not None:
                    merged.append((current_item, current_total_count))
                current_item = item
                current_total_count = count
            else:
                current_total_count += count
        
        if current_item is not None:
            merged.append((current_item, current_total_count))
        
        return merged
    
    def calculate_average_of_top5(self, compressed_detections):
        if not compressed_detections:
            return 0
        
        counts = [count for _, count in compressed_detections]
        counts.sort(reverse=True)
        top_counts = counts[:5]
        
        average_top5 = sum(top_counts) / len(top_counts)
        
        logger.info(f"Top {len(top_counts)} counts: {top_counts}")
        logger.info(f"Average of top {len(top_counts)}: {average_top5:.2f}")
        
        return average_top5
    
    def calculate_dynamic_threshold(self, compressed_detections):
        if not compressed_detections:
            return 3
        
        average_top5 = self.calculate_average_of_top5(compressed_detections)
        
        if average_top5 > 0:
            dynamic_threshold = int(average_top5 * self.percent_for_final_threshold)
            dynamic_threshold = max(3, dynamic_threshold)
            logger.info(f"Dynamic threshold calculated: {dynamic_threshold}")
            return dynamic_threshold
        
        return 3
    
    def apply_recursive_filter(self, compressed_detections):
        if not compressed_detections:
            return []
        
        dynamic_threshold = self.calculate_dynamic_threshold(compressed_detections)
        
        current_threshold = 2
        current_data = compressed_detections.copy()
        
        while current_threshold <= dynamic_threshold:
            filtered_data = []
            for item, count in current_data:
                if count >= current_threshold:
                    filtered_data.append((item, count))
            
            grouped_data = self.merge_consecutive_same(filtered_data)
            current_data = grouped_data
            current_threshold += 1
        
        return current_data
    
    def extract_interpretation(self, response_text):
        patterns = [
            r'INTERPRETATION:\s*(.+?)(?:\n|$)',
            r'Interpretation:\s*(.+?)(?:\n|$)',
            r'FINAL INTERPRETATION:\s*(.+?)(?:\n|$)',
            r'Final Interpretation:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return response_text.strip()
    
    def ask_gemini(self, filtered_format):
        if self.gemini_model is None:
            return "Gemini API key not configured"
        
        prompt = f"""
        (full prompt unchanged)
        FILTERED DATA: {filtered_format}
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            interpretation = self.extract_interpretation(response_text)
            return interpretation
            
        except Exception as e:
            return f"Error: {e}"
    
    def run_detection(self, video_path=None):
        if video_path:
            self.setup_video(video_path)
        
        self.detection_history = []
        
        self.current_letter = None
        self.current_letter_start_frame = None
        self.current_letter_count = 0
        self.current_letter_confidences = []
        
        frame_count = 0
        processed_count = 0
        
        logger.info("Starting video processing...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if (frame_count - 1) % self.GAP != 0:
                continue
            
            processed_count += 1
            
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=0.45,
                imgsz=640,
                verbose=False,
                max_det=20
            )
            
            best_class, confidence = self.get_top_detection(results)
            
            if best_class is not None:
                self.detection_history.append((processed_count, best_class, confidence))
                self.log_consecutive_detection(frame_count, best_class, confidence)
            else:
                if self.current_letter is not None:
                    self.log_consecutive_detection(frame_count, None, confidence)
        
        
        self.cap.release()
        logger.info("Processing complete.")
        return self.get_ai_interpretation()
    
    def get_ai_interpretation(self):
        if not self.detection_history:
            return "No signs detected"
        
        compressed = self.compress_consecutive_detections()
        filtered_result = self.apply_recursive_filter(compressed)
        
        logger.info(f"Final filtered list: {filtered_result}")
        
        filtered_format = " ".join([f"({letter},{count})" for letter, count in filtered_result]) if filtered_result else "No results"
        
        interpretation = self.ask_gemini(filtered_format)
        logger.info(f"AI Interpretation: {interpretation}")
        
        return interpretation


detector = ASLVideoDetector(
    model_path=MODEL_PATH,
    gap=GAP,
    conf_threshold=CONF_THRESHOLD
)

app = FastAPI(title="ASL Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        
        return JSONResponse(content={"translation": interpretation})
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
