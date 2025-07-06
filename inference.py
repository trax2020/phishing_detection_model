import os
from typing import Dict, Tuple
import numpy as np
from screenshot_capture import ScreenshotCapture
from deep_feature_extractor import DeepFeatureExtractor
from compact_feature_extractor import CompactFeatureExtractor
from cnn_trainer import CNNTrainer
from rf_trainer import RFTrainer

class PhishingDetector:
    def __init__(self, cnn_model_path: str, rf_model_path: str,
                 screenshot_dir: str = "screenshots"):
        """
        Initialize the phishing detector.
        
        Args:
            cnn_model_path (str): Path to the saved CNN model
            rf_model_path (str): Path to the saved RF model
            screenshot_dir (str): Directory to save screenshots
        """
        self.screenshot_dir = screenshot_dir
        self.screenshotter = ScreenshotCapture(screenshot_dir)
        self.deep_extractor = DeepFeatureExtractor()
        self.compact_extractor = CompactFeatureExtractor()
        
        # Load models
        self.cnn_trainer = CNNTrainer()
        self.cnn_trainer.load_model(cnn_model_path)
        
        self.rf_trainer = RFTrainer()
        self.rf_trainer.load_model(rf_model_path)
        
    def process_url(self, url: str) -> Dict:
        """
        Process a URL and return predictions from both models.
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Dictionary containing predictions and intermediate results
        """
        try:
            # Step 1: Capture screenshot
            screenshot_path = self.screenshotter.capture_screenshot(url)
            if not screenshot_path:
                raise ValueError("Failed to capture screenshot")
                
            # Step 2: Extract deep features
            deep_features = self.deep_extractor.extract_features(screenshot_path)
            if deep_features is None:
                raise ValueError("Failed to extract deep features")
                
            # Step 3: Extract compact features
            compact_features = self.compact_extractor.extract_features(screenshot_path)
            if compact_features is None:
                raise ValueError("Failed to extract compact features")
                
            # Step 4: Get CNN prediction
            cnn_prob = self.cnn_trainer.model.predict(
                deep_features.reshape(1, -1)
            )[0][0]
            
            # Step 5: Get RF prediction
            combined_features = np.concatenate([
                compact_features['scd'],
                compact_features['cld']
            ])
            rf_prob = self.rf_trainer.model.predict_proba(
                combined_features.reshape(1, -1)
            )[0][1]
            
            # Prepare results
            results = {
                'url': url,
                'screenshot_path': screenshot_path,
                'predictions': {
                    'cnn': {
                        'probability': float(cnn_prob),
                        'class': 'phishing' if cnn_prob > 0.5 else 'legitimate'
                    },
                    'rf': {
                        'probability': float(rf_prob),
                        'class': 'phishing' if rf_prob > 0.5 else 'legitimate'
                    }
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return None
            
    def cleanup(self) -> None:
        """Clean up resources"""
        self.screenshotter.close()

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = PhishingDetector(
        cnn_model_path="best_cnn_model.h5",
        rf_model_path="best_rf_model.joblib"
    )
    
    # Example URL
    url = "https://example.com"
    results = detector.process_url(url)
    
    if results:
        print("\nAnalysis Results:")
        print(f"URL: {results['url']}")
        print(f"Screenshot saved at: {results['screenshot_path']}")
        print("\nCNN Model Prediction:")
        print(f"Probability: {results['predictions']['cnn']['probability']:.4f}")
        print(f"Class: {results['predictions']['cnn']['class']}")
        print("\nRandom Forest Model Prediction:")
        print(f"Probability: {results['predictions']['rf']['probability']:.4f}")
        print(f"Class: {results['predictions']['rf']['class']}")
        
    # Cleanup
    detector.cleanup() 