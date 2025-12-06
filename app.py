from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import os
import logging
from detector import ASLVideoDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "./model/sign-detection.pt"
GAP = 3
CONF_THRESHOLD = 0.5

detector = ASLVideoDetector(
    model_path=MODEL_PATH,
    gap=GAP,
    conf_threshold=CONF_THRESHOLD
)

app = FastAPI(title="ASL Translator API")

# ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):
        return JSONResponse(content={"error": "Invalid file type. Please upload a video file."}, status_code=400)
    
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        interpretation = detector.run_detection(temp_file_path)
        
        return JSONResponse(content={"translation": interpretation})
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)