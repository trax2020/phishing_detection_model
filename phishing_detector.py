import os
import time
from typing import Dict, Optional
from screenshot_capture import ScreenshotCapture
from deep_feature_extractor import DeepFeatureExtractor
from compact_feature_extractor import CompactFeatureExtractor
from cnn_trainer import CNNTrainer
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

class PhishingDetector:
    def __init__(self, cnn_model_path: str, rf_model_path: str):
        """
        Initialize the phishing detector with all necessary components.
        
        Args:
            cnn_model_path (str): Path to the trained CNN model
            rf_model_path (str): Path to the trained Random Forest model
        """
        # Initialize screenshot capture
        self.screenshot_capture = ScreenshotCapture()
        
        # Initialize feature extractors
        self.deep_feature_extractor = DeepFeatureExtractor()
        self.compact_feature_extractor = CompactFeatureExtractor()
        
        # Initialize models
        self.cnn_trainer = CNNTrainer()
        self.cnn_trainer.load_model(cnn_model_path)
        
        self.rf_trainer = RandomForestClassifier()
        self.rf_trainer = joblib.load(rf_model_path)
        
    def process_url(self, url: str) -> Optional[Dict]:
        """
        Process a URL through the complete phishing detection pipeline.
        
        Args:
            url (str): URL to analyze
            
        Returns:
            Optional[Dict]: Dictionary containing prediction results and metadata
        """
        try:
            # Capture screenshot
            screenshot_path = self.screenshot_capture.capture_screenshot(url)
            if not screenshot_path or not os.path.exists(screenshot_path):
                return None
                
            # Extract features
            start_time = time.time()
            deep_features = self.deep_feature_extractor.extract_features(screenshot_path)
            compact_features = self.compact_feature_extractor.extract_features(screenshot_path)
            # 🔍 Debug prints to check what these contain:
            feature_extraction_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            cnn_prob = self.cnn_trainer.model.predict(np.array([deep_features]))[0][0]
            rf_prob = self.rf_trainer.predict_proba(np.array([compact_features]))[0][1]
            prediction_time = time.time() - start_time
            
            # Prepare results
            results = {
                'screenshot_path': screenshot_path,
                'feature_extraction_time': f"{feature_extraction_time:.2f}s",
                'prediction_time': f"{prediction_time:.2f}s",
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
            
    def cleanup(self):
        """Clean up resources and temporary files"""
        self.screenshot_capture.cleanup()
        self.deep_feature_extractor.cleanup()
        self.compact_feature_extractor.cleanup() 