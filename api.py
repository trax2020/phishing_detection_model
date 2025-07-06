from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Optional
import uvicorn
from phishing_detector import PhishingDetector
from fusion_engine import FusionEngine
import os
import json
from datetime import datetime

app = FastAPI(
    title="Phishing Detection API",
    description="API for detecting phishing websites using CNN and Random Forest models",
    version="1.0.0"
)

# Initialize components
detector = PhishingDetector(
    cnn_model_path="best_cnn_model.h5",
    rf_model_path="best_rf_model.joblib"
)
fusion = FusionEngine(method='weighted')

class URLRequest(BaseModel):
    url: HttpUrl
    debug: Optional[bool] = False

class PredictionResponse(BaseModel):
    url: str
    timestamp: str
    final_prediction: str
    confidence: float
    model_predictions: Dict[str, Dict[str, float]]
    debug_info: Optional[Dict] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict_phishing(request: URLRequest):
    """
    Analyze a URL for potential phishing.

    Args:
        request: URLRequest containing the URL to analyze and debug flag

    Returns:
        PredictionResponse with analysis results
    """
    try:
       # Process URL
        results = detector.process_url(str(request.url))
        if not results:
            raise HTTPException(status_code=500, detail="Failed to process URL")

        # Extract individual model predictions
        cnn_prob = results['predictions']['cnn']['probability']
        cnn_class = results['predictions']['cnn']['class']
        rf_prob = results['predictions']['rf']['probability']
        rf_class = results['predictions']['rf']['class']

        # Print individual model results
        print(f"🧠 CNN Prediction: {cnn_class} (Probability: {cnn_prob:.4f})")
        print(f"🌲 RF Prediction: {rf_class} (Probability: {rf_prob:.4f})")

        # Fuse predictions
        fused_results = fusion.fuse_predictions(cnn_prob, rf_prob)

        # Print final fused result
        print(f"🔮 FusionEngine Final Prediction: {fused_results['class']} (Confidence: {fused_results['confidence']:.4f})")

        # Helper to convert label to numeric
        def label_to_numeric(label: str) -> float:
            return 1.0 if label.lower() == 'phishing' else 0.0

        # Prepare response
        response = {
            'url': str(request.url),
            'timestamp': datetime.now().isoformat(),
            'final_prediction': fused_results['class'],
            'confidence': fused_results['confidence'],
            'model_predictions': {
                'cnn': {
                    'probability': results['predictions']['cnn']['probability'],
                    'class': label_to_numeric(results['predictions']['cnn']['class'])
                },
                'rf': {
                    'probability': results['predictions']['rf']['probability'],
                    'class': label_to_numeric(results['predictions']['rf']['class'])
                }
            }
        }

        # Add debug info if requested
        if request.debug:
            response['debug_info'] = {
                'screenshot_path': results['screenshot_path'],
                'feature_extraction_time': results.get('feature_extraction_time', 'N/A'),
                'prediction_time': results.get('prediction_time', 'N/A'),
                'brand_info': fused_results['brand_info']
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check the health of the API and models.
    
    Returns:
        dict: Health status information
    """
    try:
        # Check if models are loaded
        if not detector.cnn_trainer.model or not detector.rf_trainer.model:
            raise Exception("Models not loaded")
            
        return {
            "status": "healthy",
            "models_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )

def save_prediction_log(response: Dict):
    """
    Save prediction results to a log file.
    
    Args:
        response (dict): Prediction response to log
    """
    log_file = "prediction_logs.json"
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Load existing logs
    try:
        with open(f"logs/{log_file}", "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
        
    # Add new log entry
    logs.append(response)
    
    # Save updated logs
    with open(f"logs/{log_file}", "w") as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 