import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

class ASLClassifier:
    """
    ASL Sign Language Classifier using trained YOLO model
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier with a trained model
        Args:
            model_path: Path to the trained model weights
        """
        # Default model paths to try
        src_dir = Path(__file__).parent.absolute()
        
        if model_path:
            possible_paths = [model_path]
        else:
            possible_paths = [
                str(src_dir / "../model/retrained_asl_model.pt"),
                str(src_dir / "../../model/retrained_asl_model.pt"),
                "./model/retrained_asl_model.pt",
                "../model/retrained_asl_model.pt",
                "model/retrained_asl_model.pt",
                "./retrained_asl_model.pt",
                "retrained_asl_model.pt"
            ]
        
        self.model_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                self.model_path = abs_path
                break
        
        if self.model_path is None:
            raise FileNotFoundError(
                f"ASL model not found. Searched paths:\n" +
                "\n".join(f"  - {os.path.abspath(p)}" for p in possible_paths) +
                f"\n\nCurrent working directory: {os.getcwd()}"
            )
        
        print(f"📦 Loading model from: {os.path.basename(self.model_path)}")
        print(f"📁 Full path: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def predict_single_image(self, image_path, show=False, save=False, save_path="./prediction.jpg"):
        """
        Classify a single image
        Args:
            image_path: Path to the image file
            show: Whether to display the result (ignored in console version)
            save: Whether to save the result (ignored in console version)  
            save_path: Path to save the result (ignored in console version)
        Returns:
            dict: Prediction results with top predictions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Run prediction
        results = self.model(image_path)
        result = results[0]
        
        # Extract predictions
        probs = result.probs
        top5_indices = probs.top5
        top5_conf = probs.top5conf.tolist()
        class_names = result.names
        
        # Build prediction dictionary
        predictions = {
            'top_class': class_names[top5_indices[0]],
            'top_confidence': top5_conf[0],
            'top5': [
                {
                    'class': class_names[idx],
                    'confidence': conf
                }
                for idx, conf in zip(top5_indices, top5_conf)
            ]
        }
        
        return predictions
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'model_name': os.path.basename(self.model_path),
            'classes': list(self.model.names.values()) if hasattr(self.model, 'names') else None
        }