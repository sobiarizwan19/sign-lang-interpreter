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
import re

MODEL_PATH = "./model/sign-detection.pt"
GAP = 3
CONF_THRESHOLD = 0.5
GEMINI_API_KEY = "AIzaSyCv2XlAHLKQBCp6TzGk1GDiGLJ-EJ0mJ_g"
GEMINI_MODEL = "gemini-2.5-flash"
NO_HAND_CONFIDENCE_THRESHOLD = 0.2

# Filtering parameters
PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD = 0.3  # 30% of average of top 5 counts

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
            # If previous letter ended, log it
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

            # Reset trackers for new letter
            self.current_letter = letter
            self.current_letter_start_frame = frame_num
            self.current_letter_count = 1
            self.current_letter_confidences = [confidence]

        else:
            # Same letter, continue count
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
        """Calculate the average of the top 5 count values"""
        if not compressed_detections:
            return 0
        
        # Extract all counts
        counts = [count for _, count in compressed_detections]
        
        # Sort counts in descending order
        counts.sort(reverse=True)
        
        # Take the top 5 or all available if less than 5
        top_counts = counts[:5]
        
        # Calculate average
        average_top5 = sum(top_counts) / len(top_counts)
        
        logger.info(f"Top {len(top_counts)} counts: {top_counts}")
        logger.info(f"Average of top {len(top_counts)}: {average_top5:.2f}")
        
        return average_top5
    
    def calculate_dynamic_threshold(self, compressed_detections):
        """Calculate dynamic threshold as a percentage of the average of top 5 counts"""
        if not compressed_detections:
            return 3  # Minimum value (2 + 1)
        
        # Calculate average of top 5 counts
        average_top5 = self.calculate_average_of_top5(compressed_detections)
        
        if average_top5 > 0:
            # Calculate percentage of the average top 5
            dynamic_threshold = int(average_top5 * self.percent_for_final_threshold)
            
            # Ensure threshold is at least 3 (2 + 1)
            dynamic_threshold = max(3, dynamic_threshold)
            
            logger.info(f"Dynamic threshold calculated: {dynamic_threshold} ({self.percent_for_final_threshold*100}% of average top 5: {average_top5:.2f})")
            return dynamic_threshold
        
        return 3  # Minimum value (2 + 1)
    
    def apply_recursive_filter(self, compressed_detections):
        if not compressed_detections:
            return []
        
        # Calculate dynamic threshold based on average of top 5 counts
        dynamic_threshold = self.calculate_dynamic_threshold(compressed_detections)
        
        current_threshold = 2  # Hardcoded initial threshold
        current_data = compressed_detections.copy()
        
        while current_threshold <= dynamic_threshold:
            filtered_data = []
            for item, count in current_data:
                if count >= current_threshold:
                    filtered_data.append((item, count))
            
            grouped_data = self.merge_consecutive_same(filtered_data)
            current_data = grouped_data
            current_threshold += 1  # Hardcoded increment of 1
        
        return current_data
    
    def extract_interpretation(self, response_text):
        """Extract interpretation from Gemini response by looking for INTERPRETATION: pattern"""
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
        
        # If no pattern found, return the entire response
        return response_text.strip()
    
    def ask_gemini(self, filtered_format):
        if self.gemini_model is None:
            return "Gemini API key not configured"
        
        prompt = f"""
        I have ASL (American Sign Language) detection results in format (letter, count).
        Each (letter, count) pair represents a single letter held for "count" consecutive frames.
        The sequence of pairs represents the order of letters in the ASL message.
        When no hand is detected, it is represented as "SPACE", indicating separation between words.

        IMPORTANT NOTE ABOUT DATA QUALITY:
        The detection data comes from a computer vision model and may contain ERRORS.
        Misdetections may occur due to:
        1. Model confusion between visually similar signs
        2. False positives (detecting a letter instead of SPACE)
        3. False negatives (missing letters)
        4. Hand movements causing temporary misclassification
        5. Lighting, angle, or occlusion issues

        Your task:
        ➡ You MUST return your final interpretation in this exact format:
        INTERPRETATION: [your interpretation here]

        ➡ The interpretation must be a valid and meaningful English word, phrase, or sentence.
        ➡ It must be something a real person would logically say.
        ➡ Prefer interpretations using ONLY the detected letters.
        ➡ "SPACE" in the data indicates word separation.
        ➡ Do NOT invent unnecessary extra letters.

        IMPORTANT RULES:
        1. Use primarily the letters that appear in the filtered data.
        2. If a letter does NOT appear at all, do NOT assume it unless absolutely necessary.
        3. Respect the sequence and grouping of letters shown.
        4. Account for possible misdetections when interpreting.
        5. "SPACE" represents a word boundary.

        PRIORITIZING REPEATED LETTERS:
        - Letters with larger counts or multiple repeated detections are HIGH CONFIDENCE.
        - Interpretation MUST prioritize words formed using strongly repeated letters.
        - A confusion-based replacement is allowed ONLY IF:
            ✔ The detected letter is low count, weak, or isolated
            ✔ The replacement results in a more logical real-world interpretation
            ✔ The change does NOT modify or contradict strongly repeated letters
        - DO NOT replace a repeated or high-count letter merely to form a different word.
        Example:
            [('C', 15), ('O', 22), ('F', 23), ('E', 27)]
            → "coffee" is correct because F and E repeat strongly.
            → NOT "code", which wrongly replaces a reliable F with D.

        CRITICAL - NO ABBREVIATIONS:
        - The input contains only A-Z and SPACE — NO abbreviations.
        - Your interpretation MUST NOT contain abbreviations or acronyms.
        - Do NOT output things like: IDK, LOL, BRB, ASAP, FYI, etc.
        - Output only complete properly spelled English words or sentences.

        ALPHABET CONFUSIONS (USE ONLY WHEN NECESSARY):
        - Common ASL confusions include:
            M ↔ N ↔ T
            A ↔ S ↔ Y ↔ E
            O ↔ C
            D ↔ F
        - Space may also be confused with a letter.
        - Use these ONLY to clarify a meaningful interpretation, not to create random words.

        Handling Misinterpretations:
        - If the resulting sequence looks odd or incomplete:
            * Consider similar-letter confusion ONLY when logical
            * Remove accidental SPACE only if clearly noise
            * Correct repeated fragments caused by movement
        - Always choose the most plausible real-world meaningful interpretation.

        FILTERED DATA: {filtered_format}

        Your output MUST:
        - Start with: INTERPRETATION:
        - Contain ONLY the interpretation (no notes, NO explanation)
        - Use complete words (NO abbreviations)

        Remember: The data may contain errors — find the MOST logical and meaningful interpretation.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract interpretation using pattern matching
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
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)