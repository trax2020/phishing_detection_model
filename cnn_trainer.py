import numpy as np
import tf_keras as tf
from tf_keras.models import Sequential, Model
from tf_keras.layers import Dense, Dropout, BatchNormalization
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

class CNNTrainer:
    def __init__(self, input_dim: int = 4096, hidden_dims: List[int] = [1024, 512],
                 dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        Initialize the CNN trainer.
        
        Args:
            input_dim (int): Dimension of input features (VGG16 fc2 layer output)
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> None:
        """Build the MLP model for classification"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_dims[0], input_dim=self.input_dim,
                       activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for dim in self.hidden_dims[1:]:
            model.add(Dense(dim, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
            
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.metrics.AUC()]
        )
        
        self.model = model
        
    def load_features(self, features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # Load features from npz
        features_npz = np.load(features_path)
        features_dict = {key: features_npz[key] for key in features_npz.files}
        
        # Load labels
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
        features_dict = {key: features_npz[key] for key in features_npz.files}
        features = []
        labels = []
        for img_path, feature_vector in features_dict.items():
            if img_path in labels_dict:
                features.append(feature_vector)
                labels.append(labels_dict[img_path])
        
        return np.array(features), np.array(labels)

        
    def train(self, features: np.ndarray, labels: np.ndarray,
              validation_split: float = 0.2, batch_size: int = 32,
              epochs: int = 100, patience: int = 10) -> None:
        """
        Train the model.
        
        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): Binary labels (0 for legitimate, 1 for phishing)
            validation_split (float): Proportion of data to use for validation
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs
            patience (int): Patience for early stopping
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42,
            stratify=labels
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
            
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            features (np.ndarray): Feature vectors
            labels (np.ndarray): Binary labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        # Get predictions
        predictions = self.model.predict(features)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(predictions_binary.flatten() == labels),
            'confusion_matrix': confusion_matrix(labels, predictions_binary),
            'classification_report': classification_report(
                labels, predictions_binary, output_dict=True
            )
        }
        
        return metrics
        
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        if self.history is None:
            raise ValueError("No training history available")
            
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(model_path)
        
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model = tf.models.load_model(model_path)

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = CNNTrainer()
    
    # Example: Load features and labels
    features, labels = trainer.load_features(
        "features.npz",
        "normalized_labels.json"
    )
    # Train model
    trainer.train(features, labels)
    
    # Evaluate model
    metrics = trainer.evaluate(features, labels)
    print("Evaluation Metrics:")
    print(metrics['classification_report'])
    
    # Plot training history
    trainer.plot_training_history("training_history.png") 
    