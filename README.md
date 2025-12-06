# ASL Video Translation System

A FastAPI-based application that translates American Sign Language (ASL) from video files into natural English text using YOLO-based sign detection and Google Gemini AI for interpretation.

## 🎯 Features

- **Real-time ASL Detection**: Uses YOLOv8n model to detect individual sign letters from video frames
- **Intelligent Filtering**: Multi-stage filtering pipeline with dynamic thresholding to reduce noise and false positives
- **AI-Powered Interpretation**: Leverages Google Gemini API to convert detected letter sequences into meaningful English words and phrases
- **REST API**: Simple HTTP endpoint for video upload and translation
- **CORS Support**: Configured for cross-origin requests from frontend applications
- **Comprehensive Logging**: Detailed frame-by-frame detection logging for debugging and analysis

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing, or CPU-only mode supported)
- YOLOv8n model (trained on ASL dataset or custom sign detection weights)
- Google Gemini API key

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd sign-lang-interpreter
pip install -r requirements.txt
```

Or use the Makefile:

```bash
make install
```

### 2. Environment Setup

Create a `.env` file in the project root by copying `.env.example`:

```bash
cp .env.example .env
```

Fill in the required values:

```env
# Model Configuration
MODEL_PATH=./model/sign-detection.pt
GAP=3                                    # Frame skip interval for processing

# Detection Thresholds
CONF_THRESHOLD=0.5                       # Confidence threshold for detections
NO_HAND_CONFIDENCE_THRESHOLD=0.2         # Threshold below which to classify as SPACE

# Filtering Configuration
PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD=0.3  # Dynamic threshold percentage

# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Testing Configuration (Optional)
TEST_API_URL=http://127.0.0.1:8000
TEST_CONTENT_DIR=./test_videos
TEST_SPECIFIC_FILE=sample.mp4
```

### 3. Running the Server

Start the FastAPI server:

```bash
python app.py
```

Or using Makefile:

```bash
make run
```

The server will be available at `http://127.0.0.1:8000`

## 📡 API Endpoints

### POST `/translate`

Translates ASL video to English text.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Parameter**: `file` (video file)

**Supported Video Formats**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`

**Example cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/translate" \
  -F "file=@video.mp4"
```

**Response (Success):**
```json
{
  "translation": "hello world"
}
```

**Response (Error):**
```json
{
  "error": "Invalid file type. Please upload a video file."
}
```

## 🏗️ Project Architecture

### File Structure

```
asl-video-translator/
├── app.py                 # FastAPI application & server entry point
├── detector.py            # YOLO-based ASL detection logic
├── filtering.py           # Signal processing & filtering pipeline
├── gemini.py             # Google Gemini API integration
├── requirements.txt       # Project dependencies
├── Makefile              # Build & development commands
├── .env.example          # Environment configuration template
├── README.md             # This file
└── model/
    └── sign-detection.pt # YOLOv8n trained on ASL sign detection
```

### Module Overview

#### `app.py`
- FastAPI application initialization
- CORS middleware configuration
- `/translate` endpoint handler
- File upload validation and processing

#### `detector.py`
- `ASLVideoDetector` class
- Video frame extraction and processing at configurable intervals
- YOLOv8n model inference on each frame
- Consecutive frame detection logging and tracking

#### `filtering.py`
- `FilteringEngine` class
- Consecutive detection compression
- Recursive noise filtering with dynamic thresholds
- Average-of-top-5 threshold calculation

#### `gemini.py`
- `GeminiInterpreter` class
- Google Gemini API client initialization
- Prompt engineering for ASL interpretation
- Response parsing with pattern matching

## 🔄 Processing Pipeline

```
1. Video Upload
   ↓
2. Frame Extraction (Every Nth frame based on GAP)
   ↓
3. YOLO Detection (Confidence filtering)
   ↓
4. Consecutive Detection Compression
   ↓
5. Recursive Filtering (Dynamic threshold)
   ↓
6. Gemini AI Interpretation
   ↓
7. Response Return
```

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_PATH` | `./model/sign-detection.pt` | Path to YOLO model |
| `GAP` | `3` | Process every Nth frame (speeds up processing) |
| `CONF_THRESHOLD` | `0.5` | Minimum confidence for valid detections |
| `NO_HAND_CONFIDENCE_THRESHOLD` | `0.2` | Threshold below which to mark as SPACE |
| `PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD` | `0.3` | Dynamic threshold as % of top-5 average |

## 🧪 Testing

Run the test suite:

```bash
make test
```

This executes `test.py` which includes API endpoint testing and validation.

## 📊 Detection Quality Factors

The system accounts for common computer vision challenges:

- **Model Confusion**: Similar-looking signs (M↔N↔T, A↔S↔Y↔E, O↔C, D↔F)
- **False Positives**: Random hand gestures detected as letters
- **False Negatives**: Missed letters due to occlusion or lighting
- **Movement Artifacts**: Temporary misclassification during hand movement
- **Environmental Issues**: Lighting variations, viewing angles

The Gemini AI uses these factors to find the most plausible real-world interpretation.

## 🎓 How It Works

1. **Detection Phase**: YOLO model processes video frames at specified intervals, detecting individual ASL letter signs
2. **Compression Phase**: Consecutive identical detections are compressed into (letter, count) pairs
3. **Filtering Phase**: Multi-stage recursive filter removes noise using dynamic thresholds based on detection frequency
4. **Interpretation Phase**: Filtered letter sequences are sent to Gemini AI which interprets them as meaningful English sentences

## 🔧 Advanced Usage

### Adjusting Sensitivity

- **Increase detection sensitivity**: Lower `CONF_THRESHOLD`
- **Reduce false positives**: Increase `CONF_THRESHOLD`
- **Process fewer frames**: Increase `GAP` (faster but less accurate)
- **Process more frames**: Decrease `GAP` (slower but more accurate)

### Tuning Filtering

- **More aggressive filtering**: Increase `PERCENT_OF_AVERAGE_TOP5_FOR_FINAL_FILTER_THRESHOLD`
- **Less aggressive filtering**: Decrease the percentage value

## 📦 Dependencies

Key dependencies:
- `fastapi` - Web framework
- `ultralytics` - YOLO model inference
- `opencv-python` - Video processing
- `google-generativeai` - Gemini API
- `python-multipart` - File upload handling
- `uvicorn` - ASGI server

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### Model Loading Error
```
FileNotFoundError: Video not found or model not found
```
**Solution**: Ensure MODEL_PATH points to correct location and video file exists

### CORS Issues
```
CORSError: Cross-Origin Request Blocked
```
**Solution**: CORS is configured to allow all origins. If issues persist, check browser console

### Gemini API Error
```
Error: Gemini API key not configured
```
**Solution**: Verify GEMINI_API_KEY is set in `.env` file

### GPU Memory Issues
```
CUDA out of memory
```
**Solution**: Increase `GAP` to process fewer frames, or use CPU-only mode

## 📝 Logging

The application provides detailed logging at `INFO` level:

```
2025-01-15 10:30:45 - Loading model: ./model/sign-detection.pt
2025-01-15 10:30:47 - Starting video processing...
2025-01-15 10:30:49 - Frame     0-    5 | Detected H       | Count:  6
2025-01-15 10:30:50 - Frame     6-   11 | Detected E       | Count:  6
2025-01-15 10:31:02 - Final filtered list: [('H', 6), ('E', 6), ('L', 8), ('L', 8), ('O', 7)]
2025-01-15 10:31:05 - AI Interpretation: hello
2025-01-15 10:31:05 - Processing complete.
```

## 🚦 Performance Tips

1. Use GPU for faster inference (default)
2. Adjust `GAP` to balance speed vs. accuracy
3. Pre-process videos to consistent resolution
4. Use `.mp4` format for better compatibility
5. Monitor API response times with logging

## 🔐 Security Considerations

- Keep `GEMINI_API_KEY` secure (never commit to version control)
- Validate file types on both client and server
- Implement rate limiting for production deployment
- Use HTTPS for API endpoints in production
- Consider authentication for API access

## 📜 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs for detailed error messages
3. Verify `.env` configuration
4. Test with sample videos

## 🎬 Example Workflow

```python
# 1. Start server
# make run

# 2. Upload video
curl -X POST "http://127.0.0.1:8000/translate" \
  -F "file=@asl_video.mp4"

# 3. Receive translation
# Response: {"translation": "hello world"}
```

## 📚 References

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenCV Documentation](https://opencv.org/)

---

**Version**: 1.0.0  
**Last Updated**: December 2024