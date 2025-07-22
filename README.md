📸🔒 Image-Based Phishing Detection
Overview
This project detects phishing websites using screenshots.
It uses:
CNN: Extracts deep visual features and classifies directly
SCD (Scalable Color Descriptor): Features are classified using Random Forest (RF)
Results from CNN and RF are fused via weighted fusion or stacking fusion for final prediction.


🛠️ Tech Stack(TOp Modules Used)
Python 3.9+
TensorFlow / Keras (CNN)
Scikit-learn (Random Forest)
OpenCV (Image Processing)
Flask or FastAPI (optional API server)


📂 Project Files
api.py                 # API for predictions
cnn_model.h5           # Pre-trained CNN model
rf_model.pkl           # Pre-trained RF model on SCD features
feature_extractor.py   # Extracts CNN, SCD features
fusion.py              # Fusion logic: weighted or stacking
requirements.txt       # Dependencies
utils.py               # Helper functions
screenshots/           # Sample images


🖼️ How It Works
Input URL → Screenshot captured
CNN extracts features & predicts
SCD features → Random Forest predicts
Fusion Layer combines CNN and RF outputs
Returns final Phishing or Legitimate result with probability


📥 Setup
git clone https://github.com/yourusername/image-phishing-detector.git
cd image-phishing-detector
pip install -r requirements.txt


Run API:
python api.py
Example API call:
http://127.0.0.1:5000/predict?url=https://example.com
result = predict_url("https://example.com")
print(result)


Model Could be further retrained:
Train CNN;
python train_cnn.py

Train Random Forest on SCD;
python train_rf.py


📊 Evaluation
Accuracy: 98%
Precision: 97%
Recall: 98%
F1 Score: 99.8%
