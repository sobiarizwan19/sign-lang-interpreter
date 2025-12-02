import os
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class ASLClassifier:
    """
    ASL Sign Language Classifier using trained YOLO model
    """

    def __init__(self, model_path="./retrained_asl_model.pt"):
        """
        Initialize the classifier with a trained model

        Args:
            model_path: Path to the trained model weights
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully!")

    def predict_single_image(self, image_path, show=False, save=False, save_path="./prediction.jpg"):
        """
        Classify a single image
        
        Args:
            image_path: Path to the image file
            show: Whether to display the result
            save: Whether to save the result
            save_path: Path to save the result

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

        # Visualize if requested
        if show or save:
            self._visualize_prediction(image_path, predictions, show=show, save=save, save_path=save_path)

        return predictions

    def _visualize_prediction(self, image_path, predictions, show=True, save=False, save_path="./prediction.jpg"):
        """
        Visualize prediction results
        """
        # Load image
        img = Image.open(image_path)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f"Prediction: {predictions['top_class']}\nConfidence: {predictions['top_confidence']:.2%}",
                     fontsize=14, fontweight='bold')

        # Display top 5 predictions as bar chart
        classes = [p['class'] for p in predictions['top5']]
        confidences = [p['confidence'] for p in predictions['top5']]

        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, confidences, color='steelblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])

        # Add confidence values on bars
        for i, v in enumerate(confidences):
            ax2.text(v + 0.02, i, f'{v:.2%}', va='center')

        plt.tight_layout()

        if save:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()