import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import joblib
from typing import Dict, List, Tuple, Union, Generator
import matplotlib.pyplot as plt
import seaborn as sns
import ijson
from tqdm import tqdm

class RFTrainer:
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 random_state: int = 42):
        """Initialize the RandomForestClassifier parameters."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_importances_ = None

    def train(self, features: np.ndarray, labels: np.ndarray,
              validation_split: float = 0.2, grid_search: bool = False) -> None:
        """Train the RandomForestClassifier on the dataset."""
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=self.random_state,
            stratify=labels
        )

        if grid_search:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        self.feature_importances_ = self.model.feature_importances_

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Evaluate the classifier on test data and return performance metrics."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        metrics = {
            'accuracy': np.mean(predictions == labels),
            'auc': roc_auc_score(labels, probabilities),
            'confusion_matrix': confusion_matrix(labels, predictions),
            'classification_report': classification_report(
                labels, predictions, output_dict=True
            )
        }
        return metrics

    def plot_feature_importances(self, save_path: str = None) -> None:
        """Plot and optionally save feature importances as an image file."""
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available")

        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importances_)[::-1]
        plt.bar(range(len(self.feature_importances_)),
                self.feature_importances_[indices])
        plt.title('Feature Importances')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_model(self, model_path: str) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, model_path)

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model from disk."""
        self.model = joblib.load(model_path)

def stream_features_in_batches(features_path: str, labels_path: str,
                               batch_size: int = 200,
                               checkpoint_dir: str = "checkpoints") -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Stream features in batches and save each batch to disk (checkpointing).
    Yields only batches that have NOT yet been saved to disk.
    """
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(features_path, 'r') as f_feat, open(labels_path, 'r') as f_label:
        features_parser = ijson.kvitems(f_feat, '')  # Parse JSON key-value pairs
        labels_dict = json.load(f_label)  # Labels are expected to fit in memory

        batch_features, batch_labels = [], []
        batch_idx = 0  # Batch number counter

        for img_path, feature_dict in tqdm(features_parser, desc="Streaming features"):
            if img_path in labels_dict:
                # Convert Decimal to float to avoid type issues
                scd = [float(x) for x in feature_dict['scd']]
                cld = [float(x) for x in feature_dict['cld']]
                combined_features = np.concatenate([scd, cld])

                batch_features.append(combined_features)
                batch_labels.append(labels_dict[img_path])

                # Check if this batch should be saved (checkpointing)
                if len(batch_features) >= batch_size:
                    checkpoint_path = os.path.join(checkpoint_dir, f"batch_{batch_idx}.npz")
                    if not os.path.exists(checkpoint_path):
                        # Save the batch to disk if not already saved
                        np.savez(checkpoint_path,
                                 features=np.array(batch_features),
                                 labels=np.array(batch_labels))
                        yield np.array(batch_features), np.array(batch_labels), batch_idx
                    else:
                        print(f"Skipping existing checkpoint: batch_{batch_idx}.npz")

                    batch_features, batch_labels = [], []
                    batch_idx += 1  # Move to next batch number

        # Save any remaining data as the last batch
        if batch_features:
            checkpoint_path = os.path.join(checkpoint_dir, f"batch_{batch_idx}.npz")
            if not os.path.exists(checkpoint_path):
                np.savez(checkpoint_path,
                         features=np.array(batch_features),
                         labels=np.array(batch_labels))
                yield np.array(batch_features), np.array(batch_labels), batch_idx
            else:
                print(f"Skipping existing checkpoint: batch_{batch_idx}.npz")

if __name__ == "__main__":
    trainer = RFTrainer()
    checkpoint_dir = "checkpoints"

    batch_generator = stream_features_in_batches(
        "mini_phishing_features.json",
        "normalized_labels_windows.json",
        batch_size=200,
        checkpoint_dir=checkpoint_dir
    )

    # Process only unseen batches, others are skipped due to checkpointing
    for X_batch, y_batch, batch_idx in batch_generator:
        print(f"Processed and saved batch {batch_idx}")

    # After all batches have been streamed, load all batches from disk
    all_features, all_labels = [], []
    for filename in sorted(os.listdir(checkpoint_dir)):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(checkpoint_dir, filename))
            all_features.append(data['features'])
            all_labels.append(data['labels'])

    # Stack all saved batches into full dataset
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)

    # Train and evaluate model
    trainer.train(features, labels, grid_search=False)

    metrics = trainer.evaluate(features, labels)
    print("Evaluation Metrics:")
    print(metrics['classification_report'])

    trainer.plot_feature_importances("feature_importances.png")
    trainer.save_model("random_forest_model.joblib")
    print("Model saved as 'random_forest_model.joblib'")
    
    import numpy as np

    # Reconstruct feature names: first 20 are SCD, rest are CLD
    num_features = len(trainer.feature_importances_)
    scd_names = [f"SCD_{i}" for i in range(20)]
    cld_names = [f"CLD_{i}" for i in range(num_features - 20)]
    feature_names = scd_names + cld_names

    # Get sorted indices of features by importance
    sorted_idx = np.argsort(trainer.feature_importances_)[::-1]

    print("\nTop 20 Most Important Features:")
    for idx in sorted_idx[:20]:
        name = feature_names[idx]
        importance = trainer.feature_importances_[idx]
        print(f"{name}: {importance:.5f}")