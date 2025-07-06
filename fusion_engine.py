import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple, Optional
import joblib
import os

class FusionEngine:
    def __init__(self, method: str = 'weighted', weights: Optional[Dict[str, float]] = None,
                 meta_classifier_path: Optional[str] = None):
        """
        Initialize the fusion engine.
        
        Args:
            method (str): Fusion method ('weighted', 'average', or 'stacking')
            weights (dict): Weights for weighted voting (if method='weighted')
            meta_classifier_path (str): Path to saved meta-classifier (if method='stacking')
        """
        self.method = method
        self.weights = weights or {'cnn': 0.6, 'rf': 0.4}
        self.meta_classifier = None
        self.meta_classifier_path = meta_classifier_path
        
        if method == 'stacking' and meta_classifier_path and os.path.exists(meta_classifier_path):
            self.meta_classifier = joblib.load(meta_classifier_path)
            
    def train_meta_classifier(self, cnn_probs: np.ndarray, rf_probs: np.ndarray,
                            labels: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Train the meta-classifier for stacking.
        
        Args:
            cnn_probs (np.ndarray): CNN model probabilities
            rf_probs (np.ndarray): RF model probabilities
            labels (np.ndarray): True labels
            save_path (str): Path to save the trained meta-classifier
        """
        if self.method != 'stacking':
            raise ValueError("Meta-classifier training only supported for stacking method")
            
        # Prepare features for meta-classifier
        X_meta = np.column_stack([cnn_probs, rf_probs])
        
        # Train meta-classifier
        self.meta_classifier = LogisticRegression()
        self.meta_classifier.fit(X_meta, labels)
        
        # Save meta-classifier if path provided
        if save_path:
            joblib.dump(self.meta_classifier, save_path)
            self.meta_classifier_path = save_path
            
    def fuse_predictions(self, cnn_prob: float, rf_prob: float) -> Dict:
        """
        Fuse predictions from CNN and RF models.
        
        Args:
            cnn_prob (float): CNN model probability
            rf_prob (float): RF model probability
            
        Returns:
            dict: Fused prediction results
        """
        if self.method == 'average':
            # Simple average voting
            fused_prob = (cnn_prob + rf_prob) / 2
            
        elif self.method == 'weighted':
            # Weighted voting
            fused_prob = (self.weights['cnn'] * cnn_prob + 
                         self.weights['rf'] * rf_prob)
            
        elif self.method == 'stacking':
            if self.meta_classifier is None:
                raise ValueError("Meta-classifier not trained or loaded")
                
            # Use meta-classifier for prediction
            X_meta = np.array([[cnn_prob, rf_prob]])
            fused_prob = self.meta_classifier.predict_proba(X_meta)[0][1]
            
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")
            
        # Determine class and confidence
        is_phishing = fused_prob > 0.5
        confidence = fused_prob if is_phishing else 1 - fused_prob
        
        # Identify potential brand targeting (example implementation)
        brand_info = self._identify_brand(cnn_prob, rf_prob)
        
        return {
            'probability': float(fused_prob),
            'class': 'phishing' if is_phishing else 'legitimate',
            'confidence': float(confidence),
            'model_contributions': {
                'cnn': float(cnn_prob),
                'rf': float(rf_prob)
            },
            'brand_info': brand_info
        }
        
    def _identify_brand(self, cnn_prob: float, rf_prob: float) -> Dict:
        """
        Identify potential brand being targeted (example implementation).
        This is a placeholder - in practice, you would implement more sophisticated
        brand detection logic.
        
        Args:
            cnn_prob (float): CNN model probability
            rf_prob (float): RF model probability
            
        Returns:
            dict: Brand identification information
        """
        # Example implementation - in practice, you would:
        # 1. Extract text from the screenshot
        # 2. Compare with known brand logos/names
        # 3. Use visual similarity to known brand websites
        return {
            'detected': False,
            'brand_name': None,
            'confidence': 0.0,
            'method': 'Not implemented'
        }
        
    def save_meta_classifier(self, save_path: str) -> None:
        """
        Save the trained meta-classifier.
        
        Args:
            save_path (str): Path to save the meta-classifier
        """
        if self.meta_classifier is None:
            raise ValueError("No meta-classifier to save")
            
        joblib.dump(self.meta_classifier, save_path)
        self.meta_classifier_path = save_path

# Example usage
if __name__ == "__main__":
    # Initialize fusion engine
    fusion = FusionEngine(method='weighted')
    
    # Example predictions
    cnn_prob = 0.85
    rf_prob = 0.72
    
    # Fuse predictions
    results = fusion.fuse_predictions(cnn_prob, rf_prob)
    
    print("\nFusion Results:")
    print(f"Final Probability: {results['probability']:.4f}")
    print(f"Class: {results['class']}")
    print(f"Confidence: {results['confidence']:.4f}")
    print("\nModel Contributions:")
    print(f"CNN: {results['model_contributions']['cnn']:.4f}")
    print(f"RF: {results['model_contributions']['rf']:.4f}") 