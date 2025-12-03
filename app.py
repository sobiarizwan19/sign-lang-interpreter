#!/usr/bin/env python3

import os
import sys
import tempfile
import asyncio  # ADD THIS IMPORT
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
# Modern Sleek UI Design
# ============================================================================

# Custom CSS for modern design
modern_css = """
/* Modern gradient background */
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Card styling */
.sleek-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

/* Upload area styling */
.upload-area {
    border: 3px dashed rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
    border-radius: 16px;
    padding: 40px 20px;
    background: rgba(255, 255, 255, 0.8);
}

.upload-area:hover {
    border-color: #667eea;
    background: rgba(255, 255, 255, 0.9);
}

/* Progress bar styling */
.progress-bar {
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: rgba(102, 126, 234, 0.1);
}

.progress-bar .q-linear-progress__track {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 4px;
}

/* Result card styling */
.result-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-left: 4px solid #667eea;
    animation: fadeIn 0.6s ease-out;
}

/* Animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Confidence indicator */
.confidence-high { color: #10b981; }
.confidence-medium { color: #f59e0b; }
.confidence-low { color: #ef4444; }

/* Typography */
.display-text {
    font-size: 2.5rem;
    font-weight: 600;
    line-height: 1.3;
    color: #1f2937;
}

/* Upload button styling */
.upload-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 600;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.upload-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
}
"""

@ui.page('/')
async def main_page():
    """Modern sleek main application page - Showing ONLY AI Interpretation"""
    
    # Inject custom CSS
    ui.add_head_html(f'<style>{modern_css}</style>')
    
    # State variables
    results = {'data': None}
    processing = {'active': False}
    
    # Main container with gradient background
    with ui.column().classes('w-full min-h-screen items-center justify-center p-4 md:p-8'):
        
        # Title section
        with ui.column().classes('w-full max-w-4xl items-center text-center mb-8 md:mb-12'):
            ui.label('🤟 ASL Video Analyzer').classes('text-4xl md:text-5xl font-bold text-white mb-2')
            ui.label('Powered by YOLO + Gemini AI').classes('text-lg md:text-xl text-white/80')
        
        # Main content area
        with ui.column().classes('w-full max-w-4xl gap-6 md:gap-8'):
            
            # Upload card
            with ui.card().classes('sleek-card w-full p-6 md:p-8'):
                ui.label('Upload ASL Video').classes('text-2xl font-bold text-gray-800 mb-6 text-center')
                
                # Upload container
                upload_container = ui.column().classes('w-full items-center')
                
                # Progress indicator (hidden by default)
                progress_container = ui.column().classes('w-full items-center gap-4')
                with progress_container:
                    progress_bar = ui.linear_progress(value=0).classes('progress-bar w-full max-w-md')
                    progress_bar.visible = False
                    status_label = ui.label('').classes('text-gray-600 font-medium')
                    status_label.visible = False
                
                # File upload area
                with upload_container:
                    ui.upload(
                        on_upload=lambda e: handle_upload(e, progress_bar, status_label, results),
                        max_files=1,
                        auto_upload=True
                    ).props('''
                        accept="video/*"
                        label="📁 Drop ASL video here or click to browse"
                        color="primary"
                    ''').classes('upload-area w-full max-w-md').style('width: 100%')
                
                # Upload instructions
                ui.label('Supported formats: MP4, AVI, MOV, WebM').classes('text-sm text-gray-500 mt-4 text-center')
            
            # Results card (hidden initially)
            results_card = ui.card().classes('result-card w-full p-6 md:p-8')
            results_card.visible = False
            
            # Initialize results container
            with results_card:
                results_container = ui.column().classes('w-full items-center gap-6')
            
            async def handle_upload(e, progress_bar, status_label, results):
                """Handle video upload and processing"""
                if processing['active']:
                    ui.notify('Already processing a video', type='warning')
                    return
                
                print("\nUpload event received")
                
                try:
                    # NiceGUI 3.2.0: e.file has async read() method
                    content = await e.file.read()
                    filename = e.file.name
                    
                    print(f"Final extracted - filename: {filename}, content size: {len(content) if content else 0}")
                    
                    if not content:
                        ui.notify('No file content received', type='negative')
                        return
                    
                    # Save uploaded file
                    upload_dir = Path('./uploads')
                    upload_dir.mkdir(exist_ok=True)
                    video_path = upload_dir / (filename or 'uploaded_video.mp4')
                    
                    with open(video_path, 'wb') as f:
                        f.write(content)
                    
                    ui.notify(f'📤 Uploaded: {filename}', type='positive', position='top')
                    
                    # Show progress
                    processing['active'] = True
                    progress_bar.visible = True
                    status_label.visible = True
                    status_label.set_text('Analyzing video frames...')
                    
                    def update_progress(percent):
                        progress_bar.set_value(percent / 100)
                        status_label.set_text(f'Processing: {percent:.1f}%')
                    
                    # Process video
                    result = process_video(str(video_path), update_progress)
                    
                    # Complete progress
                    progress_bar.set_value(1.0)
                    status_label.set_text('Processing complete!')
                    
                    # Store results
                    results['data'] = result
                    
                    # Hide progress after delay
                    async def hide_progress():
                        await asyncio.sleep(1)
                        progress_bar.visible = False
                        status_label.visible = False
                    
                    # Display results
                    display_results(result, results_container, results_card)
                    
                    ui.notify('✅ Analysis complete!', type='positive', position='top')
                    
                    # Reset progress
                    await hide_progress()
                    
                except Exception as e:
                    ui.notify(f'❌ Error: {str(e)}', type='negative')
                    print(f"Upload error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    processing['active'] = False
            
            def display_results(result, container, card):
                """Display ONLY AI interpretation results"""
                card.visible = True
                container.clear()
                
                with container:
                    # Error handling
                    if 'error' in result:
                        with ui.column().classes('items-center gap-4'):
                            ui.icon('error', size='xl', color='red').classes('text-red-500')
                            ui.label(f'Error: {result["error"]}').classes('text-lg font-medium text-red-600 text-center')
                        return
                    
                    # Main AI Interpretation (ONLY SHOWING THIS)
                    with ui.column().classes('items-center gap-6 w-full'):
                        # Title
                        ui.label('🤖 AI Interpretation').classes('text-2xl font-bold text-gray-800')
                        
                        # Interpretation text in beautiful display
                        interpretation_text = result.get('interpretation', 'No interpretation available')
                        
                        with ui.card().classes('bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-2xl w-full max-w-2xl border-0'):
                            ui.label(interpretation_text).classes('display-text text-center text-gray-800')
                        
                        # Confidence indicator (minimal)
                        confidence = result.get('confidence', 'N/A')
                        confidence_class = ''
                        if 'HIGH' in str(confidence).upper():
                            confidence_class = 'confidence-high'
                        elif 'MEDIUM' in str(confidence).upper():
                            confidence_class = 'confidence-medium'
                        elif 'LOW' in str(confidence).upper():
                            confidence_class = 'confidence-low'
                        
                        if confidence != 'N/A':
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('insights', color='primary')
                                ui.label(f'Confidence: {confidence}').classes(f'text-sm font-semibold {confidence_class}')
                        
                        # Processing complete indicator
                        with ui.row().classes('items-center gap-2 mt-4'):
                            ui.icon('check_circle', color='green', size='sm')
                            ui.label('Analysis complete').classes('text-sm text-gray-500')

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