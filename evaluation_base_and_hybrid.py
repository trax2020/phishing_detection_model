import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Tuple, Dict
from cnn_trainer import CNNTrainer
from rf_trainer import RFTrainer
from fusion_engine import FusionEngine


class Evaluator:
    def __init__(self):
        self.metrics = {}

    def load_cnn_data(self, features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        features_npz = np.load(features_path)
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
        features, labels = [], []
        for key in features_npz.files:
            if key in labels_dict:
                features.append(features_npz[key])
                labels.append(labels_dict[key])
        return np.array(features), np.array(labels)

    def load_rf_data(self, features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(features_path, 'r') as f:
            features_dict = json.load(f)
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)

        features, labels = [], []
        for key, feat in features_dict.items():
            if key in labels_dict:
                combined_features = feat['scd'] + feat['cld']  # Flatten SCD + CLD
                features.append(combined_features)
                labels.append(labels_dict[key])
        return np.array(features), np.array(labels)

    def evaluate_model(self, true_labels: np.ndarray, predictions: np.ndarray, model_name: str):
        binary_preds = (predictions > 0.5).astype(int)
        try:
            auc = roc_auc_score(true_labels, predictions)
        except ValueError:
            auc = None
        self.metrics[model_name] = {
            'accuracy': accuracy_score(true_labels, binary_preds),
            'precision': precision_score(true_labels, binary_preds, zero_division=0),
            'recall': recall_score(true_labels, binary_preds, zero_division=0),
            'f1_score': f1_score(true_labels, binary_preds, zero_division=0),
            'auc': auc,
            'confusion_matrix': confusion_matrix(true_labels, binary_preds).tolist()
        }

    def plot_individual_roc(self, true_labels: np.ndarray, preds: np.ndarray, model_name: str):
        plt.figure(figsize=(6,5))
        try:
            fpr, tpr, _ = roc_curve(true_labels, preds)
            auc = roc_auc_score(true_labels, preds)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
        except ValueError:
            print(f"Skipping ROC for {model_name} due to single class in y_true.")
            return
        plt.plot([0,1], [0,1], 'k--')
        plt.title(f'{model_name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{model_name}_roc_curve.png")
        plt.show()

    def plot_individual_conf_matrix(self, model_name: str):
        plt.figure(figsize=(4,4))
        metric = self.metrics[model_name]
        sns.heatmap(
            np.array(metric['confusion_matrix']),
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Phish'],
            yticklabels=['Legit', 'Phish']
        )
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{model_name}_conf_matrix.png")
        plt.show()

    def plot_combined_roc(self, true_labels_dict: Dict[str, np.ndarray], preds_dict: Dict[str, np.ndarray]):
        plt.figure(figsize=(8,6))
        for model_name in preds_dict:
            try:
                fpr, tpr, _ = roc_curve(true_labels_dict[model_name], preds_dict[model_name])
                auc = roc_auc_score(true_labels_dict[model_name], preds_dict[model_name])
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
            except ValueError:
                print(f"Skipping ROC for {model_name} due to single class in y_true.")
        plt.plot([0,1], [0,1], 'k--')
        plt.title('Combined ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig("combined_roc_curve.png")
        plt.show()

    def generate_report(self, save_path: str = None) -> str:
        report = "Evaluation Report\n====================\n\n"
        for model_name, metric in self.metrics.items():
            report += f"{model_name} Model:\n"
            report += f"Accuracy : {metric['accuracy']:.4f}\n"
            report += f"Precision: {metric['precision']:.4f}\n"
            report += f"Recall   : {metric['recall']:.4f}\n"
            report += f"F1 Score : {metric['f1_score']:.4f}\n"
            report += f"AUC      : {metric['auc'] if metric['auc'] else 'N/A'}\n\n"
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        return report


if __name__ == "__main__":
    evaluator = Evaluator()

    # CNN Evaluation
    cnn = CNNTrainer()
    cnn.load_model("best_cnn_model.h5")
    cnn_features, cnn_labels = evaluator.load_cnn_data("features.npz", "labels for cnn.json")
    cnn_preds = cnn.model.predict(cnn_features).flatten()
    evaluator.evaluate_model(cnn_labels, cnn_preds, "CNN")

    # RF Evaluation
    rf = RFTrainer()
    rf.load_model("best_rf_model.joblib")
    rf_features, rf_labels = evaluator.load_rf_data("mini_phishing_features.json", "labels for rf.json")
    rf_preds = rf.model.predict_proba(rf_features)[:, 1]
    evaluator.evaluate_model(rf_labels, rf_preds, "RandomForest")

    # Hybrid Fusion Evaluation
    from fusion_engine import FusionEngine
    fusion = FusionEngine(method='weighted')

    # Apply fusion to CNN + RF outputs
    fused_probs = [
        fusion.fuse_predictions(cnn_p, rf_p)['probability']
        for cnn_p, rf_p in zip(cnn_preds, rf_preds)
    ]
    fused_probs = np.array(fused_probs)
    evaluator.evaluate_model(cnn_labels, fused_probs, "HybridFusion")

    # Plot Individual ROC and Confusion Matrix for CNN
    evaluator.plot_individual_roc(cnn_labels, cnn_preds, "CNN")
    evaluator.plot_individual_conf_matrix("CNN")

    # Plot Individual ROC and Confusion Matrix for RF
    evaluator.plot_individual_roc(rf_labels, rf_preds, "RandomForest")
    evaluator.plot_individual_conf_matrix("RandomForest")

    # Plot Individual ROC and Confusion Matrix for Hybrid Fusion
    evaluator.plot_individual_roc(cnn_labels, fused_probs, "HybridFusion")
    evaluator.plot_individual_conf_matrix("HybridFusion")

    # Plot Combined ROC Curve
    true_labels_dict = {'CNN': cnn_labels, 'RandomForest': rf_labels, 'HybridFusion': cnn_labels}
    preds_dict = {'CNN': cnn_preds, 'RandomForest': rf_preds, 'HybridFusion': fused_probs}
    evaluator.plot_combined_roc(true_labels_dict, preds_dict)

    # Report
    print(evaluator.generate_report("hybrid_evaluation_report.txt"))
