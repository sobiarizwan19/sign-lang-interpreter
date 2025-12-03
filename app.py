#!/usr/bin/env python3

import os
import sys
import tempfile
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
# NiceGUI Web Interface
# ============================================================================

@ui.page('/')
async def main_page():
    """Main application page"""
    
    # State variables
    results = {'data': None}
    processing = {'active': False}
    
    # Header
    with ui.header().classes('items-center justify-between'):
        ui.label('🤟 ASL Video Analyzer').classes('text-2xl font-bold')
        ui.label('Powered by YOLO + Gemini AI').classes('text-sm opacity-70')
    
    # Main container
    with ui.column().classes('w-full max-w-4xl mx-auto p-6 gap-4'):
        
        # Upload section
        with ui.card().classes('w-full p-6'):
            ui.label('Upload ASL Video').classes('text-xl font-semibold mb-4')
            
            upload_container = ui.column().classes('w-full')
            progress_bar = ui.linear_progress(value=0).classes('w-full')
            progress_bar.visible = False
            status_label = ui.label('').classes('mt-2')
            status_label.visible = False
            
            async def handle_upload(e):
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
                    
                    ui.notify(f'Processing {filename}...', type='info')
                    
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
                    
                    # Update progress
                    progress_bar.set_value(1.0)
                    status_label.set_text('Processing complete!')
                    
                    # Store results
                    results['data'] = result
                    
                    # Display results
                    display_results(result)
                    
                    ui.notify('Analysis complete!', type='positive')
                    
                except Exception as e:
                    ui.notify(f'Error: {str(e)}', type='negative')
                    print(f"Upload error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    processing['active'] = False
                    progress_bar.visible = False
            
            # File uploader
            with upload_container:
                ui.upload(
                    on_upload=handle_upload,
                    max_files=1,
                    auto_upload=True
                ).props('accept=video/*').classes('w-full')
        
        # Results section
        with ui.card().classes('w-full p-6') as results_card:
            results_card.visible = False
            results_container = ui.column().classes('w-full gap-4')
        
        def display_results(result):
            """Display analysis results"""
            results_card.visible = True
            results_container.clear()
            
            with results_container:
                # Title
                ui.label('📊 Analysis Results').classes('text-xl font-semibold mb-2')
                
                # Error handling
                if 'error' in result:
                    ui.label(f'❌ Error: {result["error"]}').classes('text-red-600')
                    return
                
                # Detected sequence
                with ui.card().classes('w-full bg-blue-50 p-4'):
                    ui.label('🔤 Detected Letter Sequence:').classes('font-semibold mb-2')
                    ui.label(result['sequence'] or 'No signs detected').classes('text-2xl font-mono')
                
                # AI Interpretation (MAIN RESULT)
                with ui.card().classes('w-full bg-green-50 p-4'):
                    ui.label('🤖 AI Interpretation:').classes('font-semibold mb-2')
                    ui.label(result['interpretation']).classes('text-3xl font-bold text-green-700')
                    
                    # Confidence and reasoning
                    with ui.row().classes('gap-4 mt-2'):
                        ui.label(f'Confidence: {result["confidence"]}').classes('text-sm font-semibold')
                    
                    if result.get('reasoning'):
                        ui.label(f'Reasoning: {result["reasoning"]}').classes('text-sm mt-2 opacity-70')
                
                # Alternative interpretations
                if result.get('alternatives'):
                    with ui.card().classes('w-full bg-yellow-50 p-4'):
                        ui.label('💡 Alternative Interpretations:').classes('font-semibold mb-2')
                        for alt in result['alternatives']:
                            ui.label(f'• {alt}').classes('text-sm')
                
                # Statistics
                with ui.card().classes('w-full bg-gray-50 p-4'):
                    ui.label('📈 Statistics:').classes('font-semibold mb-2')
                    stats = f"""
                    Total Frames: {result.get('total_frames', 'N/A')}
                    Processed Frames: {result.get('processed_frames', 'N/A')}
                    Detected Signs: {result.get('detected_signs', 'N/A')}
                    Frame Gap: {FRAME_GAP}
                    """
                    ui.label(stats).classes('text-sm whitespace-pre-line')

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
        show=False
    )