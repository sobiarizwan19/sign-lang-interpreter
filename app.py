#!/usr/bin/env python3

import os
import sys
import tempfile
import asyncio
import cv2
from pathlib import Path
from nicegui import ui, app
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import ASL modules
try:
    from src.signPredict import ASLClassifier
    from src.predictSentence import SentencePredictor
    print("✓ All ASL components imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Configuration
VIDEO_PATH = os.getenv('VIDEO_PATH', '../content/demo.mp4')
FRAME_GAP = int(os.getenv('FRAME_GAP', '10'))
MODEL_PATH = os.getenv('MODEL_PATH', '../model/retrained_asl_model.pt')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

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
            print(f"✓ Gemini predictor initialized: {sentence_predictor.model_name}")
        else:
            print("⚠ Warning: GEMINI_API_KEY not found. AI interpretation will be limited.")
            sentence_predictor = None
            
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        raise

def process_video(video_path: str, progress_callback=None):
    """
    Process video and extract ASL sequence with AI interpretation
    
    Args:
        video_path: Path to the video file
        progress_callback: Optional callback for progress updates
        
    Returns:
        dict: Results including sequence and interpretation
    """
    if not os.path.exists(video_path):
        return {
            'error': f'Video file not found: {video_path}',
            'sequence': '',
            'interpretation': '',
            'confidence': 'LOW'
        }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'error': f'Cannot open video: {video_path}',
            'sequence': '',
            'interpretation': '',
            'confidence': 'LOW'
        }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = {}
    sequence = []
    frame_count = 0
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_count % FRAME_GAP == 0:
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)
            
            try:
                # Get prediction
                result = classifier.predict_single_image(temp_path, show=False, save=False)
                
                if result and result['top_confidence'] > CONFIDENCE_THRESHOLD:
                    predictions[frame_count] = result
                    
                    # Add to sequence if different from last
                    pred_letter = result['top_class']
                    if not sequence or sequence[-1] != pred_letter:
                        sequence.append(pred_letter)
                        
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            # Progress callback
            if progress_callback:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
        
        frame_count += 1
    
    cap.release()
    
    # Build sequence string
    sequence_str = " ".join(sequence)
    
    # Get AI interpretation
    interpretation_result = None
    if sentence_predictor and sequence_str:
        try:
            interpretation_result = sentence_predictor.predict_sentence(sequence_str)
        except Exception as e:
            print(f"Error getting interpretation: {e}")
            interpretation_result = {
                'interpretation': sequence_str,
                'confidence': 'LOW',
                'reasoning': f'Error: {str(e)}',
                'alternatives': []
            }
    
    # Return results
    return {
        'sequence': sequence_str,
        'interpretation': interpretation_result['interpretation'] if interpretation_result else sequence_str,
        'confidence': interpretation_result['confidence'] if interpretation_result else 'N/A',
        'reasoning': interpretation_result.get('reasoning', '') if interpretation_result else '',
        'alternatives': interpretation_result.get('alternatives', []) if interpretation_result else [],
        'total_frames': total_frames,
        'processed_frames': len(predictions),
        'detected_signs': len(sequence)
    }

# ============================================================================
# Minimal Modern UI Design
# ============================================================================

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
                
                # Process video
                result = process_video(str(video_path), update_progress)
                
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
            """Display only the AI interpretation"""
            area.visible = True
            container.clear()
            
            with container:
                # Error handling
                if 'error' in result:
                    ui.label(f'Error: {result["error"]}').classes('text-red-500 text-center')
                    return
                
                # Display AI interpretation
                interpretation_text = result.get('interpretation', 'No interpretation available')
                ui.label(interpretation_text).classes('ai-output-text')

# ============================================================================
# Application Startup
# ============================================================================

if __name__ in {'__main__', '__mp_main__'}:
    print("🚀 Starting ASL Video Analyzer...")
    
    # Initialize models
    init_models()
    
    # Create uploads directory
    Path('./uploads').mkdir(exist_ok=True)
    
    print("🌟 Application ready! Open your browser to start analyzing ASL videos.")
    
    # Start NiceGUI
    ui.run(
        title='ASL Video Analyzer',
        port=8080,
        reload=False,
        show=False,
        favicon='🤟'
    )