#!/usr/bin/env python3
"""
ASL Video Analyzer - NiceGUI Application
A beautiful, self-contained application for ASL video analysis
"""

import os
import sys
import asyncio
import tempfile
import uuid
import traceback
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add src directory to path for imports
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / 'src'
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# NiceGUI imports
from nicegui import ui
from nicegui.events import UploadEventArguments
import nicegui

# Try to import ASL components
COMPONENTS_AVAILABLE = False
try:
    from signPredict import ASLClassifier
    from predictSentence import SentencePredictor
    from asl_console_analyzer import ASLConsoleAnalyzer
    COMPONENTS_AVAILABLE = True
    print("✓ All ASL components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ASL components: {e}")
    print("Make sure signPredict.py, predictSentence.py, and asl_console_analyzer.py are in the src/ directory")

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
UPLOAD_DIR = Path('uploads')
UPLOAD_DIR.mkdir(exist_ok=True)

class ASLAnalyzerApp:
    def __init__(self):
        self.current_file_path: Optional[str] = None
        self.analysis_result: Optional[Dict[str, Any]] = None
        
        # Load configuration
        self.config = self._load_config()
        
        # UI components
        self.upload_area = None
        self.progress_bar = None
        self.result_card = None
        self.status_label = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment or defaults"""
        config = {
            'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY', 'AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs'),
            'GEMINI_MODEL': os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
            'MODEL_PATH': os.getenv('MODEL_PATH', 'model/retrained_asl_model.pt'),
            'CONFIDENCE_THRESHOLD': float(os.getenv('CONFIDENCE_THRESHOLD', '0.5')),
            'FRAME_GAP': int(os.getenv('FRAME_GAP', '10'))
        }
        
        # Find model file
        model_paths = [
            config['MODEL_PATH'],
            'model/retrained_asl_model.pt',
            '../model/retrained_asl_model.pt',
            './retrained_asl_model.pt',
            'retrained_asl_model.pt'
        ]
        
        for path in model_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                config['MODEL_PATH'] = abs_path
                break
        else:
            config['MODEL_PATH'] = None
            
        return config
    
    def create_ui(self):
        """Create the main user interface"""
        # Custom CSS for beautiful styling
        ui.add_head_html('''
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@400;700&display=swap');
        
        :root {
            --primary-color: #6366f1;
            --secondary-color: #ec4899;
            --accent-color: #10b981;
            --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --card-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --glass-effect: rgba(255, 255, 255, 0.1);
        }
        
        body {
            background: var(--background-gradient);
            min-height: 100vh;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .main-container {
            backdrop-filter: blur(10px);
            background: var(--glass-effect);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: var(--card-shadow);
        }
        
        .hero-title {
            font-family: 'Playfair Display', serif;
            background: linear-gradient(45deg, #6366f1, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            color: rgba(255, 255, 255, 0.8);
            text-align: center;
            font-weight: 400;
            margin-bottom: 2rem;
        }
        
        .upload-zone {
            border: 3px dashed rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            position: relative;
        }
        
        .upload-zone:hover {
            border-color: var(--primary-color);
            background: rgba(99, 102, 241, 0.1);
            transform: translateY(-2px);
        }
        
        .upload-icon {
            font-size: 4rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .result-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            box-shadow: var(--card-shadow);
            padding: 2rem;
            margin-top: 2rem;
            color: #1f2937;
        }
        
        .result-header {
            font-family: 'Playfair Display', serif;
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        
        .sequence-display {
            font-family: 'JetBrains Mono', monospace;
            background: linear-gradient(45deg, #f3f4f6, #e5e7eb);
            padding: 1rem;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.1em;
            border-left: 4px solid var(--primary-color);
        }
        
        .interpretation-display {
            background: linear-gradient(45deg, #ecfdf5, #d1fae5);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
            font-weight: 600;
            margin-top: 1rem;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 0.25rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .confidence-high { background: #10b981; color: white; }
        .confidence-medium { background: #f59e0b; color: white; }
        .confidence-low { background: #ef4444; color: white; }
        
        .progress-container {
            margin: 2rem 0;
        }
        
        .status-text {
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            margin-top: 1rem;
            font-weight: 500;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }
        
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1rem;
        }
        
        .component-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.875rem;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        
        .status-success { background: #10b981; }
        .status-error { background: #ef4444; }
        </style>
        ''')
        
        # Main container
        with ui.column().classes('w-full max-w-4xl mx-auto p-6'):
            # Header
            ui.html('<h1 class="hero-title text-5xl">ASL Video Analyzer</h1>', sanitize=False)
            ui.html('<p class="subtitle text-lg">Advanced American Sign Language Recognition powered by AI</p>', sanitize=False)
            
            # System status
            with ui.card().classes('main-container p-6 mb-6'):
                ui.label('System Status').classes('text-white text-lg font-semibold mb-4')
                
                # Component status indicators
                with ui.row().classes('gap-4'):
                    with ui.column():
                        status_html = self._get_status_html()
                        ui.html(status_html, sanitize=False)
            
            # Main application area
            with ui.card().classes('main-container p-8'):
                # Upload area
                self.upload_area = self._create_upload_area()
                
                # Progress bar (initially hidden)
                with ui.column().classes('progress-container').style('display: none') as self.progress_container:
                    ui.label('Processing video...').classes('text-white text-center font-semibold')
                    self.progress_bar = ui.linear_progress(value=0).classes('w-full')
                    self.status_label = ui.label('').classes('status-text')
                
                # Results area (initially hidden)
                self.result_container = ui.column().classes('w-full').style('display: none')
                
    def _get_status_html(self) -> str:
        """Generate system status HTML"""
        components_status = "success" if COMPONENTS_AVAILABLE else "error"
        components_text = "Available" if COMPONENTS_AVAILABLE else "Not Available"
        
        model_status = "success" if self.config.get('MODEL_PATH') and os.path.exists(self.config['MODEL_PATH']) else "error"
        model_text = "Found" if model_status == "success" else "Not Found"
        
        return f'''
        <div class="component-status">
            <div class="status-indicator status-{components_status}"></div>
            <span>ASL Components: {components_text}</span>
        </div>
        <div class="component-status">
            <div class="status-indicator status-{model_status}"></div>
            <span>AI Model: {model_text}</span>
        </div>
        <div class="component-status">
            <div class="status-indicator status-success"></div>
            <span>Gemini API: Configured</span>
        </div>
        '''
    
    def _create_upload_area(self):
        """Create the file upload area"""
        with ui.column().classes('w-full'):
            # Upload zone with button
            with ui.element('div').classes('upload-zone'):
                ui.html('<div class="upload-icon">🎥</div>', sanitize=False)
                ui.label('Upload your ASL video').classes('text-white text-xl font-semibold')
                ui.label('Click the button below to browse files').classes('text-white/60 text-sm mt-2')
                ui.label(f'Supported formats: {", ".join(ALLOWED_EXTENSIONS)}').classes('text-white/40 text-xs mt-2')
                ui.label(f'Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB').classes('text-white/40 text-xs')
                
                # Simple upload button - use a function wrapper to handle async
                def handle_upload_sync(e):
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task if loop is running
                        loop.create_task(self._handle_upload(e))
                    else:
                        # Run directly if no loop
                        asyncio.run(self._handle_upload(e))
                
                upload = ui.upload(
                    on_upload=handle_upload_sync,
                    max_file_size=MAX_FILE_SIZE,
                    multiple=False,
                    auto_upload=True,
                    label='Choose Video File'
                ).props('accept="video/*"').classes('mt-4')
            
            return upload
    
    async def _handle_upload(self, e):
        """Handle video file upload and processing"""
        try:
            print(f"Upload event received")
            print(f"Event attributes: {[attr for attr in dir(e) if not attr.startswith('_')]}")
            
            # In NiceGUI, uploaded files are in e.file
            filename = None
            content = None
            
            if hasattr(e, 'file') and e.file:
                # Get filename
                if hasattr(e.file, 'name'):
                    filename = e.file.name
                elif hasattr(e.file, 'filename'):
                    filename = e.file.filename
                
                # Get content - need to await the read() method
                if hasattr(e.file, 'read'):
                    content = await e.file.read()
                elif hasattr(e.file, 'content'):
                    content = e.file.content
                    if hasattr(content, 'read'):
                        content = await content.read()
            
            print(f"Final extracted - filename: {filename}, content size: {len(content) if content else 0}")
            
            # Default filename if none found
            if not filename:
                filename = "uploaded_video.mp4"
            
            if not content or len(content) == 0:
                self._show_error("No file content received - please try uploading again")
                return
            
            # Validate file
            if not self._validate_file(filename, content):
                return
            
            # Save uploaded file
            file_path = await self._save_upload(filename, content)
            self.current_file_path = file_path
            
            # Show progress and start analysis
            self._show_progress()
            await self._analyze_video(file_path)
            
        except Exception as error:
            self._show_error(f"Upload failed: {str(error)}")
            print(f"Upload error: {traceback.format_exc()}")
    
    def _validate_file(self, filename: str, content) -> bool:
        """Validate uploaded file"""
        try:
            file_ext = Path(filename).suffix.lower()
            
            if file_ext not in ALLOWED_EXTENSIONS:
                self._show_error(f"Invalid file type. Supported: {', '.join(ALLOWED_EXTENSIONS)}")
                return False
            
            # Check file size
            if isinstance(content, bytes):
                file_size = len(content)
            else:
                self._show_error("Invalid file content type")
                return False
                
            if file_size > MAX_FILE_SIZE:
                self._show_error(f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
                return False
                
            return True
            
        except Exception as error:
            self._show_error(f"File validation failed: {str(error)}")
            return False
    
    async def _save_upload(self, filename: str, content) -> str:
        """Save uploaded file and return path"""
        try:
            file_id = str(uuid.uuid4())
            file_ext = Path(filename).suffix.lower()
            if not file_ext:
                file_ext = '.mp4'  # Default extension
                
            new_filename = f"{file_id}{file_ext}"
            file_path = UPLOAD_DIR / new_filename
            
            with open(file_path, 'wb') as f:
                if isinstance(content, bytes):
                    f.write(content)
                else:
                    raise Exception("Content must be bytes")
                
            return str(file_path.absolute())
            
        except Exception as error:
            raise Exception(f"Failed to save file: {str(error)}")
    
    def _show_progress(self):
        """Show progress indicator"""
        self.upload_area.style('display: none')
        self.progress_container.style('display: block')
        self.result_container.style('display: none')
    
    def _show_error(self, message: str):
        """Show error message"""
        with self.result_container:
            self.result_container.clear()
            with ui.card().classes('result-card'):
                ui.html(f'<div class="error-message">❌ {message}</div>', sanitize=False)
        
        self.progress_container.style('display: none')
        self.result_container.style('display: block')
    
    async def _analyze_video(self, video_path: str):
        """Perform ASL video analysis"""
        try:
            if not COMPONENTS_AVAILABLE:
                raise Exception("ASL analysis components not available")
            
            if not self.config['MODEL_PATH'] or not os.path.exists(self.config['MODEL_PATH']):
                raise Exception("ASL model not found")
            
            # Update progress
            self.progress_bar.value = 0.1
            self.status_label.text = "Initializing analyzer..."
            await asyncio.sleep(0.1)
            
            # Create analyzer
            analyzer = ASLConsoleAnalyzer(
                video_path=video_path,
                frame_gap=self.config['FRAME_GAP'],
                model_path=self.config['MODEL_PATH'],
                gemini_api_key=self.config['GEMINI_API_KEY'],
                gemini_model=self.config['GEMINI_MODEL'],
                confidence_threshold=self.config['CONFIDENCE_THRESHOLD']
            )
            
            self.progress_bar.value = 0.3
            self.status_label.text = "Processing video frames..."
            await asyncio.sleep(0.1)
            
            # Run analysis in background
            result = await self._run_analysis(analyzer)
            
            self.progress_bar.value = 1.0
            self.status_label.text = "Analysis complete!"
            await asyncio.sleep(0.5)
            
            # Show results
            self._show_results(result)
            
        except Exception as error:
            self._show_error(f"Analysis failed: {str(error)}")
            print(f"Analysis error: {traceback.format_exc()}")
        finally:
            # Cleanup
            if self.current_file_path and os.path.exists(self.current_file_path):
                try:
                    os.remove(self.current_file_path)
                except:
                    pass
    
    async def _run_analysis(self, analyzer) -> Dict[str, Any]:
        """Run the actual video analysis"""
        import io
        import contextlib
        
        # Capture output from analyzer
        captured_output = io.StringIO()
        
        # Update progress during analysis
        async def update_progress():
            for i in range(30, 90, 10):
                await asyncio.sleep(0.5)
                self.progress_bar.value = i / 100
                if i == 40:
                    self.status_label.text = "Detecting hand signs..."
                elif i == 60:
                    self.status_label.text = "Building sequence..."
                elif i == 80:
                    self.status_label.text = "Generating interpretation..."
        
        # Start progress updates
        progress_task = asyncio.create_task(update_progress())
        
        try:
            # Run analysis with captured output
            with contextlib.redirect_stdout(captured_output):
                analyzer.analyze_video()
                
            output_text = captured_output.getvalue()
            
            # Cancel progress updates
            progress_task.cancel()
            
            # Parse output
            return self._parse_analysis_output(output_text)
            
        except Exception as e:
            progress_task.cancel()
            raise e
    
    def _parse_analysis_output(self, output_text: str) -> Dict[str, Any]:
        """Parse analyzer output into structured result"""
        lines = output_text.strip().split('\n')
        detected_sequence = ""
        interpretation = ""
        
        for line in lines:
            if line.startswith('DETECTED SEQUENCE:'):
                detected_sequence = line.replace('DETECTED SEQUENCE:', '').strip()
            elif line.startswith('INTERPRETATION:'):
                interpretation = line.replace('INTERPRETATION:', '').strip()
        
        if not detected_sequence and not interpretation:
            if "No signs detected" in output_text:
                detected_sequence = "No signs detected"
                interpretation = "No recognizable ASL signs found in the video"
            else:
                raise Exception("Failed to parse analysis output")
        
        # Determine confidence
        confidence = "HIGH"
        if detected_sequence == interpretation:
            confidence = "HIGH"
        elif "No signs" in detected_sequence or "No signs" in interpretation:
            confidence = "LOW"
        else:
            confidence = "MEDIUM"
        
        return {
            'detected_sequence': detected_sequence,
            'interpretation': interpretation,
            'confidence': confidence,
            'raw_output': output_text
        }
    
    def _show_results(self, result: Dict[str, Any]):
        """Display analysis results"""
        self.progress_container.style('display: none')
        
        with self.result_container:
            self.result_container.clear()
            
            with ui.card().classes('result-card'):
                ui.html('<h2 class="result-header">🎯 Analysis Results</h2>', sanitize=False)
                
                # Detected sequence
                ui.label('Detected Letter Sequence:').classes('font-semibold text-gray-700 mb-2')
                ui.html(f'<div class="sequence-display">{result["detected_sequence"]}</div>', sanitize=False)
                
                # Interpretation
                ui.label('Interpretation:').classes('font-semibold text-gray-700 mb-2 mt-4')
                ui.html(f'<div class="interpretation-display">{result["interpretation"]}</div>', sanitize=False)
                
                # Confidence
                with ui.row().classes('mt-4 items-center gap-2'):
                    ui.label('Confidence:').classes('font-semibold text-gray-700')
                    confidence_class = f"confidence-{result['confidence'].lower()}"
                    ui.html(f'<span class="confidence-badge {confidence_class}">{result["confidence"]}</span>', sanitize=False)
                
                # Action buttons
                with ui.row().classes('mt-6 gap-4'):
                    ui.button('Analyze Another Video', on_click=self._reset_app).classes('btn-primary')
                    
                    if result['raw_output']:
                        ui.button('View Details', on_click=lambda: self._show_details(result['raw_output'])).classes('btn-primary')
        
        self.result_container.style('display: block')
    
    def _reset_app(self):
        """Reset application for new analysis"""
        self.current_file_path = None
        self.analysis_result = None
        
        self.upload_area.style('display: block')
        self.progress_container.style('display: none')
        self.result_container.style('display: none')
        
        ui.notify('Ready for new video analysis', type='positive')
    
    def _show_details(self, raw_output: str):
        """Show detailed analysis output"""
        with ui.dialog() as dialog, ui.card():
            ui.label('Detailed Analysis Output').classes('text-lg font-bold mb-4')
            with ui.scroll_area().classes('w-96 h-64'):
                ui.code(raw_output).classes('text-xs')
            
            with ui.row().classes('mt-4'):
                ui.button('Close', on_click=dialog.close)
        
        dialog.open()

def main():
    """Main application entry point"""
    print("🚀 Starting ASL Video Analyzer...")
    
    # Create application instance
    app_instance = ASLAnalyzerApp()
    
    # Check system status
    if not COMPONENTS_AVAILABLE:
        print("⚠️  Warning: ASL components not found. Please ensure:")
        print("   - signPredict.py is in src/ directory")
        print("   - predictSentence.py is in src/ directory") 
        print("   - asl_console_analyzer.py is in src/ directory")
        print("   - Required dependencies are installed")
    
    if not app_instance.config['MODEL_PATH']:
        print("⚠️  Warning: ASL model not found. Please ensure retrained_asl_model.pt is in model/ directory")
    
    # Set up NiceGUI app
    ui.page_title = 'ASL Video Analyzer'
    
    # Create UI
    app_instance.create_ui()
    
    # Run application
    print("🌟 Application ready! Open your browser to start analyzing ASL videos.")
    ui.run(
        title='ASL Video Analyzer',
        port=8080,
        show=True,
        reload=False,
        dark=False
    )

if __name__ == '__main__':
    main()