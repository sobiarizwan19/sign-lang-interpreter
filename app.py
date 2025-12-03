#!/usr/bin/env python3
"""
ASL Video Analyzer Backend API - Fixed for src directory structure
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
from pathlib import Path
import sys
import logging
import traceback

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create uploads directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add both current directory and src directory to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / 'src'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Python path includes: {[str(current_dir), str(src_dir)]}")

# Try to import ASL components from src directory
COMPONENTS_AVAILABLE = False
try:
    from signPredict import ASLClassifier
    from predictSentence import SentencePredictor
    from asl_console_analyzer import ASLConsoleAnalyzer
    COMPONENTS_AVAILABLE = True
    print("✓ All components imported successfully from src/")
except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    print(f"Files in src/: {list(src_dir.glob('*.py')) if src_dir.exists() else 'src/ not found'}")

def load_env():
    """Load environment variables from .env file"""
    env_file = current_dir / '.env'
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Process uploaded video and return ASL interpretation"""
    
    temp_path = None
    
    try:
        print("=== Starting analyze request ===")
        
        if not COMPONENTS_AVAILABLE:
            return jsonify({
                'error': 'ASL analysis components not available. Make sure files are in src/ directory.',
                'success': False
            }), 500
        
        # Check if file is in request
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided',
                'success': False
            }), 400
        
        file = request.files['video']
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        # Check file size
        file_content = file.read()
        file_size = len(file_content)
        print(f"File size: {file_size} bytes")
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': 'File size too large (max 50MB)',
                'success': False
            }), 400
        
        file.seek(0)
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file format',
                'success': False
            }), 400
        
        # Save file
        file_id = str(uuid.uuid4())
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        temp_filename = f"{file_id}.{file_ext}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        temp_abs_path = os.path.abspath(temp_path)
        
        print(f"Saving to: {temp_path}")
        print(f"Absolute path: {temp_abs_path}")
        file.save(temp_path)
        
        # Load configuration
        env_vars = load_env()
        GEMINI_API_KEY = env_vars.get('GEMINI_API_KEY', 'AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs')
        GEMINI_MODEL = env_vars.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        MODEL_PATH = env_vars.get('MODEL_PATH', 'model/retrained_asl_model.pt')
        CONFIDENCE_THRESHOLD = float(env_vars.get('CONFIDENCE_THRESHOLD', '0.5'))
        FRAME_GAP = int(env_vars.get('FRAME_GAP', '10'))
        
        print(f"Config - Model: {MODEL_PATH}, Frame gap: {FRAME_GAP}, Confidence: {CONFIDENCE_THRESHOLD}")
        
        # Check if model file exists
        model_abs_path = os.path.abspath(MODEL_PATH)
        print(f"Looking for model at: {model_abs_path}")
        
        if not os.path.exists(model_abs_path):
            # Try alternative paths
            alt_paths = [
                'model/retrained_asl_model.pt',
                '../model/retrained_asl_model.pt',
                './retrained_asl_model.pt',
                'retrained_asl_model.pt'
            ]
            
            found_model = None
            for alt_path in alt_paths:
                abs_alt_path = os.path.abspath(alt_path)
                print(f"Trying alternative path: {abs_alt_path}")
                if os.path.exists(abs_alt_path):
                    found_model = alt_path
                    break
            
            if found_model:
                MODEL_PATH = found_model
                print(f"Found model at: {MODEL_PATH}")
            else:
                return jsonify({
                    'error': f'Model file not found. Searched: {model_abs_path} and alternatives',
                    'success': False
                }), 500
        
        print("Creating analyzer...")
        
        # Convert model path to absolute path since Flask found it
        model_abs_path = os.path.abspath(MODEL_PATH)
        print(f"Using absolute model path: {model_abs_path}")
        print(f"Using absolute video path: {temp_abs_path}")
        
        # Create analyzer - use absolute paths to avoid any path confusion
        analyzer = ASLConsoleAnalyzer(
            video_path=temp_abs_path,  # Use absolute path for video
            frame_gap=FRAME_GAP,
            model_path=model_abs_path,  # Use absolute path for model
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=GEMINI_MODEL,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        print("Running analysis...")
        
        # Run analysis and capture output
        import io
        import contextlib
        
        captured_output = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(captured_output):
                analyzer.analyze_video()
            
            output_text = captured_output.getvalue()
            print(f"Analysis completed. Output: {output_text}")
            
        except Exception as analysis_error:
            print(f"Analysis failed: {analysis_error}")
            print(f"Analysis traceback: {traceback.format_exc()}")
            
            return jsonify({
                'error': f'Video analysis failed: {str(analysis_error)}',
                'success': False
            }), 500
        
        # Parse the output
        lines = output_text.strip().split('\n')
        detected_sequence = ""
        interpretation = ""
        
        for line in lines:
            if line.startswith('DETECTED SEQUENCE:'):
                detected_sequence = line.replace('DETECTED SEQUENCE:', '').strip()
            elif line.startswith('INTERPRETATION:'):
                interpretation = line.replace('INTERPRETATION:', '').strip()
        
        print(f"Final result - Sequence: '{detected_sequence}', Interpretation: '{interpretation}'")
        
        # Handle case where no output was captured
        if not detected_sequence and not interpretation:
            return jsonify({
                'error': 'No signs detected in video or analysis failed silently',
                'success': False,
                'raw_output': output_text
            }), 500
        
        return jsonify({
            'success': True,
            'detected_sequence': detected_sequence,
            'interpretation': interpretation,
            'confidence': 'HIGH' if detected_sequence == interpretation else 'MEDIUM'
        })
    
    except Exception as e:
        print(f"=== ERROR in analyze_video ===")
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500
    
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up: {temp_path}")
            except Exception as cleanup_error:
                print(f"Failed to cleanup {temp_path}: {cleanup_error}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    env_vars = load_env()
    model_path = env_vars.get('MODEL_PATH', 'model/retrained_asl_model.pt')
    model_exists = os.path.exists(model_path)
    
    return jsonify({
        'status': 'healthy',
        'components_available': COMPONENTS_AVAILABLE,
        'model_path': model_path,
        'model_exists': model_exists,
        'current_directory': str(current_dir),
        'src_directory': str(src_dir),
        'src_files': [f.name for f in src_dir.glob('*.py')] if src_dir.exists() else [],
        'message': 'ASL Video Analyzer API is running'
    })

if __name__ == '__main__':
    print("Starting ASL Video Analyzer API...")
    print(f"Components available: {COMPONENTS_AVAILABLE}")
    
    env_vars = load_env()
    model_path = env_vars.get('MODEL_PATH', 'model/retrained_asl_model.pt')
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)