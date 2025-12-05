#!/usr/bin/env python3
"""
Simple ASL Video Test Script
Tests videos against API and matches with filenames (case insensitive)

Usage:
1. Test all videos: Set SPECIFIC_FILE = None
2. Test specific video: Set SPECIFIC_FILE = "filename.mp4"

Examples:
- SPECIFIC_FILE = None                    # Test all videos
- SPECIFIC_FILE = "good-morning.mp4"      # Test only good-morning.mp4
- SPECIFIC_FILE = "i-love-you.mp4"        # Test only i-love-you.mp4
"""

import os
import requests
import time

# Configuration
API_URL = "http://127.0.0.1:8000/translate"
CONTENT_DIR = "./content"
SPECIFIC_FILE = "dog.mp4"  # Set to filename (e.g., "good-morning.mp4") to test only that file, or None to test all files

def filename_to_expected(filename):
    """Convert filename to expected text."""
    # Remove .mp4 extension and replace hyphens with spaces
    expected = filename.replace('.mp4', '').replace('-', ' ')
    return expected.lower()

def normalize_text(text):
    """Normalize text for comparison (lowercase, strip spaces)."""
    return text.lower().strip()

def test_video(video_path):
    """Test a single video."""
    filename = os.path.basename(video_path)
    expected = filename_to_expected(filename)
    
    print(f"\n🎥 Testing: {filename}")
    print(f"Expected: '{expected}'")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'file': (filename, f, 'video/mp4')}
            response = requests.post(API_URL, files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                interpretation = result.get('interpretation', '')
                
                print(f"AI Result: '{interpretation}'")
                
                # Check match (case insensitive)
                expected_norm = normalize_text(expected)
                actual_norm = normalize_text(interpretation)
                
                if expected_norm == actual_norm:
                    print("✅ MATCH!")
                    return True
                elif expected_norm in actual_norm or actual_norm in expected_norm:
                    print("✅ PARTIAL MATCH!")
                    return True
                else:
                    print("❌ NO MATCH!")
                    return False
            else:
                print(f"❌ API Error: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all video tests."""
    print("🧪 ASL Video Tester")
    print("=" * 50)
    
    # Get video files based on configuration
    video_files = []
    
    if SPECIFIC_FILE:
        # Test only the specific file
        specific_path = os.path.join(CONTENT_DIR, SPECIFIC_FILE)
        if os.path.exists(specific_path) and SPECIFIC_FILE.endswith('.mp4'):
            video_files.append(specific_path)
            print(f"Testing specific file: {SPECIFIC_FILE}")
        else:
            print(f"❌ Specific file not found: {SPECIFIC_FILE}")
            return
    else:
        # Get all video files
        for file in os.listdir(CONTENT_DIR):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(CONTENT_DIR, file))
        video_files.sort()
        print(f"Found {len(video_files)} videos")
    
    print("=" * 50)
    
    # Test each video
    passed = 0
    total = len(video_files)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{total}]", end="")
        if test_video(video_path):
            passed += 1
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%" if total > 0 else "0.0%")
    print("=" * 50)

if __name__ == "__main__":
    # Check if API is running
    try:
        requests.get("http://127.0.0.1:8000", timeout=5)
        print("✅ API is running")
    except:
        print("❌ API not running! Start with: python app.py")
        exit(1)
    
    main()