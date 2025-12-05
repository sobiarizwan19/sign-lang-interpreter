#!/usr/bin/env python3
"""
ASL Video Test Script with Partial Word Match Percentage
Tests videos against API and calculates match percentage.

Usage:
1. Test all videos: Set SPECIFIC_FILE = None
2. Test specific video: Set SPECIFIC_FILE = "filename.mp4"
"""

import os
import requests
import time

# Configuration
API_URL = "http://127.0.0.1:8002/translate"
CONTENT_DIR = "./content"
SPECIFIC_FILE = "thank-you.mp4"  # Set to filename or None
# SPECIFIC_FILE = None  # Test all videos
def filename_to_expected(filename):
    """Convert filename to expected text."""
    expected = filename.replace('.mp4', '').replace('-', ' ')
    return expected.lower()

def normalize_text(text):
    """Normalize text for comparison (lowercase, strip spaces)."""
    return text.lower().strip()

def calculate_match_percentage(expected, actual):
    """Calculate match percentage with partial word scoring."""
    expected_words = expected.lower().split()
    actual_words = actual.lower().split()
    
    # Exact match
    if expected.lower().strip() == actual.lower().strip():
        return 100
    
    # Count matched words
    matched_count = sum(1 for word in expected_words if word in actual_words)
    
    # Partial match: proportion of words matched out of total, scaled to 80%
    if matched_count > 0:
        partial_score = (matched_count / len(expected_words)) * 80
        return round(partial_score, 1)
    
    # No words matched
    return 0

def test_video(video_path):
    """Test a single video with match percentage."""
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
                
                match_percent = calculate_match_percentage(expected, interpretation)
                
                if match_percent == 100:
                    print("✅ EXACT MATCH! (100%)")
                elif match_percent > 0:
                    print(f"✅ PARTIAL MATCH ({match_percent}%)")
                else:
                    print("❌ NO MATCH (0%)")
                
                return match_percent
            else:
                print(f"❌ API Error: {response.status_code}")
                return 0
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

def main():
    """Run all video tests."""
    print("🧪 ASL Video Tester")
    print("=" * 50)
    
    # Get video files
    video_files = []
    
    if SPECIFIC_FILE:
        specific_path = os.path.join(CONTENT_DIR, SPECIFIC_FILE)
        if os.path.exists(specific_path) and SPECIFIC_FILE.endswith('.mp4'):
            video_files.append(specific_path)
            print(f"Testing specific file: {SPECIFIC_FILE}")
        else:
            print(f"❌ Specific file not found: {SPECIFIC_FILE}")
            return
    else:
        for file in os.listdir(CONTENT_DIR):
            if file.endswith('.mp4'):
                video_files.append(os.path.join(CONTENT_DIR, file))
        video_files.sort()
        print(f"Found {len(video_files)} videos")
    
    print("=" * 50)
    
    total_percent = 0
    total = len(video_files)
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{total}]", end="")
        match_percent = test_video(video_path)
        total_percent += match_percent
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Total Videos: {total}")
    print(f"Average Success Rate: {total_percent/total:.1f}%" if total > 0 else "0.0%")
    print("=" * 50)

if __name__ == "__main__":
    # Check if API is running
    try:
        requests.get("http://127.0.0.1:8002", timeout=5)
        print("✅ API is running")
    except:
        print("❌ API not running! Start with: python app.py")
        exit(1)
    
    main()
