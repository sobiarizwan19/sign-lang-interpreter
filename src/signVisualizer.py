import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import tempfile

try:
    from signPredict import ASLClassifier
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("❌ signPredict.py not found!")

try:
    from predictSentence import SentencePredictor
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("❌ predictSentence.py not found!")

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.frame_gap = 5
        self.classifier = None
        self.sentence_predictor = None
        self.predictions = {}
        self.paused = True
        self.timer = QTimer()
        self.video_loaded = False
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Sign Language Video Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top controls
        top_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        top_layout.addWidget(self.load_btn)
        
        self.video_label = QLabel("No video loaded")
        top_layout.addWidget(self.video_label)
        
        top_layout.addStretch()
        
        # Frame gap selector
        top_layout.addWidget(QLabel("Frame Gap:"))
        self.gap_combo = QComboBox()
        self.gap_combo.addItems(["1", "2", "3", "5", "10", "15"])
        self.gap_combo.setCurrentText("5")
        self.gap_combo.currentTextChanged.connect(self.change_frame_gap)
        top_layout.addWidget(self.gap_combo)
        
        main_layout.addLayout(top_layout)
        
        # Video display area
        display_layout = QHBoxLayout()
        
        # Video display
        self.video_display = QLabel()
        self.video_display.setMinimumSize(800, 500)
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setStyleSheet("background-color: black;")
        display_layout.addWidget(self.video_display, 2)
        
        # Prediction sidebar
        sidebar = QVBoxLayout()
        
        # Current prediction
        pred_group = QGroupBox("Current Prediction")
        pred_layout = QVBoxLayout()
        
        self.pred_label = QLabel("--")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #4CAF50;")
        pred_layout.addWidget(self.pred_label)
        
        self.conf_label = QLabel("Confidence: --")
        self.conf_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.conf_label)
        
        self.frame_label = QLabel("Frame: 0/0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        pred_layout.addWidget(self.frame_label)
        
        pred_group.setLayout(pred_layout)
        sidebar.addWidget(pred_group)
        
        # Prediction history
        hist_group = QGroupBox("Recent Predictions")
        hist_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        hist_layout.addWidget(self.history_list)
        
        hist_group.setLayout(hist_layout)
        sidebar.addWidget(hist_group)
        
        # Sequence display
        seq_group = QGroupBox("Detected Sequence")
        seq_layout = QVBoxLayout()
        
        self.sequence_display = QTextEdit()
        self.sequence_display.setReadOnly(True)
        self.sequence_display.setMaximumHeight(100)
        seq_layout.addWidget(self.sequence_display)
        
        # Add AI Interpret button
        self.interpret_btn = QPushButton("🤖 Interpret with AI")
        self.interpret_btn.clicked.connect(self.interpret_with_ai)
        self.interpret_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        seq_layout.addWidget(self.interpret_btn)
        
        seq_group.setLayout(seq_layout)
        sidebar.addWidget(seq_group)
        
        # AI Interpretation display
        ai_group = QGroupBox("AI Interpretation")
        ai_layout = QVBoxLayout()
        
        self.ai_display = QTextEdit()
        self.ai_display.setReadOnly(True)
        self.ai_display.setMaximumHeight(200)
        self.ai_display.setPlaceholderText("Click 'Interpret with AI' to analyze the detected sequence...")
        ai_layout.addWidget(self.ai_display)
        
        ai_group.setLayout(ai_layout)
        sidebar.addWidget(ai_group)
        
        display_layout.addLayout(sidebar, 1)
        main_layout.addLayout(display_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        
        self.prev_btn = QPushButton("⏮ Previous Frame")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next Frame ⏭")
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        control_layout.addWidget(self.next_btn)
        
        main_layout.addLayout(control_layout)
        
        # Progress bar
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.valueChanged.connect(self.slider_changed)
        main_layout.addWidget(self.progress_slider)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Timer for playback
        self.timer.timeout.connect(self.next_frame)
        
        # Initialize classifier and LLM
        self.init_classifier()
        self.init_llm()
        
    def init_classifier(self):
        """Initialize the sign language classifier"""
        if MODEL_AVAILABLE:
            try:
                self.classifier = ASLClassifier()
                self.status_bar.showMessage("Model loaded successfully")
            except Exception as e:
                self.status_bar.showMessage(f"Model error: {str(e)}")
                self.classifier = None
        else:
            self.classifier = None
            self.status_bar.showMessage("Model not available")
    
    def init_llm(self):
        """Initialize the LLM sentence predictor"""
        if LLM_AVAILABLE:
            try:
                self.sentence_predictor = SentencePredictor()
                print("✓ LLM initialized successfully")
            except Exception as e:
                print(f"LLM initialization error: {e}")
                self.sentence_predictor = None
                self.interpret_btn.setEnabled(False)
        else:
            self.sentence_predictor = None
            self.interpret_btn.setEnabled(False)
    
    def interpret_with_ai(self):
        """Interpret the detected sequence using AI"""
        sequence = self.sequence_display.toPlainText().strip()
        
        if not sequence:
            QMessageBox.warning(self, "No Sequence", "No alphabet sequence detected yet!\n\nPlease play the video to detect signs first.")
            return
        
        if not self.sentence_predictor:
            QMessageBox.critical(self, "LLM Error", "Sentence predictor not initialized!\n\nPlease check your API key and internet connection.")
            return
        
        # Show loading message
        self.ai_display.setText("🔄 Analyzing sequence with AI...\n\nPlease wait...")
        QApplication.processEvents()  # Update UI
        
        try:
            # Call LLM
            result = self.sentence_predictor.predict_sentence(sequence)
            
            # Format and display result
            formatted_result = self.sentence_predictor.format_result(result)
            self.ai_display.setText(formatted_result)
            
            self.status_bar.showMessage(f"AI Interpretation: {result['interpretation']}")
            
        except Exception as e:
            error_msg = f"Error analyzing sequence:\n\n{str(e)}"
            self.ai_display.setText(error_msg)
            QMessageBox.critical(self, "Analysis Error", error_msg)
    
    def load_video(self):
        """Load a video file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.setText(os.path.basename(file_path))
            
            # Open video
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Cannot open video file!")
                return
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.current_frame_idx = 0
            self.predictions.clear()
            self.history_list.clear()
            self.sequence_display.clear()
            self.ai_display.clear()
            
            # Setup progress slider
            self.progress_slider.setRange(0, self.total_frames - 1)
            self.progress_slider.setEnabled(True)
            
            # Enable controls
            self.play_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            
            self.video_loaded = True
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)} - {self.total_frames} frames")
            
            # Show first frame
            self.show_frame()
    
    def show_frame(self):
        """Display the current frame with prediction"""
        if not self.video_loaded or not self.cap:
            return
        
        # Set video position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Get or create prediction (only for frames matching frame_gap)
        if self.current_frame_idx % self.frame_gap == 0:
            if self.current_frame_idx not in self.predictions:
                self.status_bar.showMessage(f"Predicting frame {self.current_frame_idx}...")
                prediction = self.predict_frame(frame)
                self.predictions[self.current_frame_idx] = prediction
                self.status_bar.showMessage(f"Frame {self.current_frame_idx}: {prediction['prediction']} ({prediction['confidence']:.1%})")
            else:
                prediction = self.predictions[self.current_frame_idx]
        else:
            # Find nearest predicted frame
            prediction = self.find_nearest_prediction()
            if prediction["prediction"] == "--":
                # Show message that this frame doesn't align with gap
                self.status_bar.showMessage(f"Frame {self.current_frame_idx} - Use Next/Prev to jump to predicted frames (gap={self.frame_gap})")
        
        # Update display
        self.update_display(frame, prediction)
        
        # Update progress slider without triggering valueChanged
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(self.current_frame_idx)
        self.progress_slider.blockSignals(False)
        
        # Update frame label
        self.frame_label.setText(f"Frame: {self.current_frame_idx}/{self.total_frames - 1}")
    
    def predict_frame(self, frame):
        """Predict sign for a frame"""
        if self.classifier is None:
            return {"prediction": "NO MODEL", "confidence": 0.0}
        
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            # Get prediction
            result = self.classifier.predict_single_image(temp_path, show=False, save=False)
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "prediction": result['top_class'],
                "confidence": float(result['top_confidence']),
                "all_predictions": result['top5']
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"prediction": "ERROR", "confidence": 0.0}
    
    def find_nearest_prediction(self):
        """Find the nearest predicted frame"""
        if not self.predictions:
            return {"prediction": "--", "confidence": 0.0}
        
        # Find the closest predicted frame
        closest_idx = min(self.predictions.keys(), 
                         key=lambda x: abs(x - self.current_frame_idx))
        
        if abs(closest_idx - self.current_frame_idx) <= self.frame_gap * 2:
            return self.predictions[closest_idx]
        else:
            return {"prediction": "--", "confidence": 0.0}
    
    def update_display(self, frame, prediction):
        """Update the display with frame and prediction"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        
        # Create QImage
        qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Create overlay with prediction
        pixmap = QPixmap.fromImage(qimage)
        painter = QPainter(pixmap)
        
        # Add prediction overlay
        pred_text = prediction["prediction"]
        confidence = prediction["confidence"]
        
        # Set color based on confidence
        if confidence > 0.7:
            color = QColor(76, 175, 80)  # Green
        elif confidence > 0.4:
            color = QColor(255, 193, 7)  # Yellow
        else:
            color = QColor(244, 67, 54)  # Red
        
        # Draw semi-transparent overlay
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, width, 80)
        
        # Draw prediction text
        font = QFont()
        font.setPointSize(36)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color, 3))
        
        text_rect = QRect(0, 0, width, 80)
        painter.drawText(text_rect, Qt.AlignCenter, pred_text)
        
        # Draw confidence
        font.setPointSize(16)
        painter.setFont(font)
        painter.setPen(QPen(Qt.white, 2))
        
        conf_text = f"{confidence:.1%}"
        conf_rect = QRect(0, 50, width, 30)
        painter.drawText(conf_rect, Qt.AlignCenter, conf_text)
        
        # Draw frame number and gap indicator
        frame_text = f"Frame {self.current_frame_idx}"
        painter.setPen(QPen(Qt.white, 1))
        painter.drawText(10, height - 10, frame_text)
        
        # Add indicator if this is a predicted frame
        if self.current_frame_idx % self.frame_gap == 0:
            painter.setPen(QPen(QColor(76, 175, 80), 2))
            painter.drawText(10, height - 30, f"✓ Predicted (Gap: {self.frame_gap})")
        else:
            painter.setPen(QPen(QColor(255, 193, 7), 2))
            painter.drawText(10, height - 30, f"⚠ Not predicted (Gap: {self.frame_gap})")
        
        painter.end()
        
        # Scale pixmap to fit display
        scaled_pixmap = pixmap.scaled(
            self.video_display.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.video_display.setPixmap(scaled_pixmap)
        
        # Update prediction labels
        self.pred_label.setText(pred_text)
        self.conf_label.setText(f"Confidence: {confidence:.1%}")
        
        # Update prediction history (only for predicted frames)
        if self.current_frame_idx % self.frame_gap == 0 and confidence > 0.3:
            history_text = f"Frame {self.current_frame_idx}: {pred_text} ({confidence:.1%})"
            self.history_list.addItem(history_text)
            self.history_list.scrollToBottom()
            
            # Update sequence display
            self.update_sequence(pred_text, confidence)
    
    def update_sequence(self, prediction, confidence):
        """Update the sequence display"""
        if confidence < 0.5 or prediction in ["--", "ERROR", "NO MODEL"]:
            return
        
        current_text = self.sequence_display.toPlainText()
        predictions = current_text.split()
        
        # Only add if different from last prediction
        if not predictions or predictions[-1] != prediction:
            predictions.append(prediction)
            new_text = " ".join(predictions)
            self.sequence_display.setText(new_text)
    
    def next_frame(self):
        """Go to next frame - respects frame_gap"""
        if not self.video_loaded:
            return
        
        # Jump by frame_gap instead of 1
        next_idx = self.current_frame_idx + self.frame_gap
        
        if next_idx < self.total_frames:
            self.current_frame_idx = next_idx
            self.show_frame()
        else:
            # Stop at last frame
            self.paused = True
            self.play_btn.setText("▶ Play")
            self.timer.stop()
    
    def prev_frame(self):
        """Go to previous frame - respects frame_gap"""
        if not self.video_loaded or self.current_frame_idx == 0:
            return
        
        # Jump back by frame_gap instead of 1
        prev_idx = self.current_frame_idx - self.frame_gap
        self.current_frame_idx = max(0, prev_idx)
        self.show_frame()
    
    def toggle_play(self):
        """Toggle play/pause"""
        if not self.video_loaded:
            return
        
        self.paused = not self.paused
        
        if self.paused:
            self.play_btn.setText("▶ Play")
            self.timer.stop()
        else:
            self.play_btn.setText("⏸ Pause")
            # Calculate delay based on frame_gap and fps
            delay = int((1000 / self.fps) * self.frame_gap)
            self.timer.start(delay)
    
    def slider_changed(self, value):
        """Handle progress slider change"""
        if not self.video_loaded:
            return
        
        self.current_frame_idx = value
        self.show_frame()
    
    def change_frame_gap(self, value):
        """Change frame gap setting"""
        old_gap = self.frame_gap
        self.frame_gap = int(value)
        
        # Snap to nearest valid frame for new gap
        if self.video_loaded:
            # Round current frame to nearest multiple of new gap
            self.current_frame_idx = (self.current_frame_idx // self.frame_gap) * self.frame_gap
            
            # Clear predictions since gap changed
            self.predictions.clear()
            self.history_list.clear()
            self.sequence_display.clear()
            self.ai_display.clear()
            
            # Refresh current frame
            self.show_frame()
        
        self.status_bar.showMessage(f"Frame gap changed to: {value} (moved to frame {self.current_frame_idx})")
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if self.cap:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application style
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
        }
        QLabel {
            color: #ffffff;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:disabled {
            background-color: #666666;
        }
        QGroupBox {
            color: #ffffff;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px;
            background: #333333;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4CAF50;
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        QListWidget {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
        }
        QTextEdit {
            background-color: #333333;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 4px;
        }
    """)
    
    player = VideoPlayer()
    player.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()