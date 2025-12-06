import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import logging
import tempfile
import shutil
from filtering import FilteringEngine
from gemini import GeminiInterpreter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CONF_THRESHOLD = 0.5
NO_HAND_CONFIDENCE_THRESHOLD = 0.2
GAP = 3


class ASLVideoDetector:
    def __init__(self, model_path, gap, conf_threshold):
        self.GAP = gap
        self.conf_threshold = conf_threshold
        self.no_hand_threshold = NO_HAND_CONFIDENCE_THRESHOLD
        
        self.filtering_engine = FilteringEngine()
        self.gemini_interpreter = GeminiInterpreter()
        
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
                    self.log_consecutive_detection(frame_num, None, confidence)
        
        
        self.cap.release()
        logger.info("Processing complete.")
        return self.get_ai_interpretation()
    
    def get_ai_interpretation(self):
        if not self.detection_history:
            return "No signs detected"
        
        compressed = self.filtering_engine.compress_consecutive_detections(self.detection_history)
        filtered_result = self.filtering_engine.apply_recursive_filter(compressed)
        
        logger.info(f"Final filtered list: {filtered_result}")
        
        filtered_format = " ".join([f"({letter},{count})" for letter, count in filtered_result]) if filtered_result else "No results"
        
        interpretation = self.gemini_interpreter.ask_gemini(filtered_format)
        logger.info(f"AI Interpretation: {interpretation}")
        
        return interpretation