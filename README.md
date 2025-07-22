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
| **File**                             | **Purpose / Description**                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **api.py**                           | Hosts the API endpoint to accept URLs, trigger screenshot capture, run detection, and return results (phishing/legitimate).                                             |
| **best\_cnn\_model.h5**              | The pre-trained CNN model saved in HDF5 format, used for deep visual feature extraction and direct classification.                                                      |
| **best\_rf\_model.joblib**           | The trained Random Forest model on SCD features, serialized with Joblib for inference use.                                                                              |
| **cnn\_trainer.py**                  | Script to train the CNN model on labeled screenshot data and save the best-performing model (`best_cnn_model.h5`).                                                      |
| **compact\_feature\_extractor.py**   | Extracts **Scalable Color Descriptor (SCD)** and potentially other compact visual features from website screenshots.                                                    |
| **deep\_feature\_extractor.py**      | Extracts deep features from screenshots using the CNN model without classifying — just feature vectors for fusion.                                                      |
| **evaluation\_base\_and\_hybrid.py** | Evaluates both **base models (CNN, RF)** and the **hybrid fused models**. Outputs metrics like accuracy, precision, recall, F1-score, and ROC.                          |
| **evaluator.py**                     | Contains utilities or functions to compute performance metrics, confusion matrix, ROC curve, etc., given predictions and ground truths.                                 |
| **fusion\_engine.py**                | Implements fusion strategies — **weighted fusion** or **stacking fusion** — to combine CNN and RF outputs for improved final predictions.                               |
| **inference.py**                     | Centralized inference logic to load models (CNN + RF), process input images, extract features, and perform prediction (with or without fusion).                         |
| **phishing\_detector.py**            | The main detection pipeline script. Likely combines all components: captures screenshot, extracts features, applies classifiers, fuses outputs, returns final decision. |
| **requirements.txt**                 | Lists Python package dependencies needed to run the project.                                                                                                            |
| **rf\_trainer.py**                   | Script to train the Random Forest model on extracted **SCD features** and save the trained model as `best_rf_model.joblib`.                                             |
| **screenshot\_capture.py**           | Captures screenshots of a provided URL, preparing image inputs for feature extraction.                                                                                  |



🖼️ How It Works
Input URL → Screenshot captured
CNN extracts features & predicts
SCD features → Random Forest predicts
Fusion Layer combines CNN and RF outputs
Returns final Phishing or Legitimate result with probability


📥 Setup
git clone https://github.com/trax2020/phishing_detection_model.git

cd image-phishing-detector

pip install -r requirements.txt


Run API:
python api.py
Example API call:
http://127.0.0.1:8000/predict?url=https://example.com
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
