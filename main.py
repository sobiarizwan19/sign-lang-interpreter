#!/usr/bin/env python3
"""
ASL Video Analyzer - Main Entry Point
Processes video files to detect ASL signs and interpret sequences
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(src_dir))

# Load environment variables
def load_env():
    """Load environment variables from .env file"""
    env_file = src_dir / '.env'
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def main():
    """Main entry point"""
    print("🚀 ASL Video Analyzer")
    print("=" * 50)
    
    # Load environment configuration
    env_vars = load_env()
    
    # Get configuration values with defaults
    video_path = env_vars.get('VIDEO_PATH', '../content/video.mp4')
    frame_gap = int(env_vars.get('FRAME_GAP', '10'))
    gemini_api_key = env_vars.get('GEMINI_API_KEY', 'AIzaSyCb-XaqhT3v1He3cTRH0zn6QCZFwHBKRNs')
    model_path = env_vars.get('MODEL_PATH', '../model/retrained_asl_model.pt')
    confidence_threshold = float(env_vars.get('CONFIDENCE_THRESHOLD', '0.5'))
    progress_frequency = int(env_vars.get('PROGRESS_UPDATE_FREQUENCY', '50'))
    
    print(f"📁 Video path: {video_path}")
    print(f"⚙️  Frame gap: {frame_gap}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print(f"🤖 AI API: {'✅ Configured' if gemini_api_key else '❌ Not configured'}")
    print()
    
    # Import and initialize analyzer
    try:
        from src.asl_console_analyzer import ASLConsoleAnalyzer
        
        # Create analyzer with configuration
        analyzer = ASLConsoleAnalyzer(
            video_path=video_path,
            frame_gap=frame_gap,
            model_path=model_path,
            gemini_api_key=gemini_api_key,
            confidence_threshold=confidence_threshold,
            progress_frequency=progress_frequency
        )
        
        # Run analysis
        analyzer.analyze_video()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required files are in the src/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()