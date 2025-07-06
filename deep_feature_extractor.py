import numpy as np
import tensorflow as tf
from tf_keras.applications.vgg16 import VGG16, preprocess_input
from tf_keras.models import Model
from tf_keras.preprocessing import image
import cv2
import os

class DeepFeatureExtractor:
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the deep feature extractor using VGG16.
        
        Args:
            target_size (tuple): Target size for image resizing (height, width)
        """
        self.target_size = target_size
        
        # Load pre-trained VGG16 model
        base_model = VGG16(weights='imagenet', include_top=True)
        
        # Create a new model that outputs the features from the last fully connected layer
        self.model = Model(inputs=base_model.input,
                          outputs=base_model.get_layer('fc2').output)
        
    def preprocess_image(self, img_path):
        """
        Preprocess the image for VGG16.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Read image using OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.target_size)
        
        # Convert to array and preprocess for VGG16
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
        
    def extract_features(self, img_path):
        """
        Extract deep features from an image.
        
        Args:
            img_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Feature vector of shape (4096,)
        """
        try:
            # Preprocess the image
            img_array = self.preprocess_image(img_path)
            
            # Extract features
            features = self.model.predict(img_array, verbose=0)
            
            # Flatten the features
            features = features.flatten()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {img_path}: {str(e)}")
            return None
            
    def extract_features_batch(self, img_paths):
        """
        Extract features from multiple images.
        
        Args:
            img_paths (list): List of paths to image files
            
        Returns:
            dict: Dictionary mapping image paths to their feature vectors
        """
        features_dict = {}
        
        for img_path in img_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                continue
                
            features = self.extract_features(img_path)
            if features is not None:
                features_dict[img_path] = features
                
        return features_dict

# Example usage
if __name__ == "__main__":
    
    # Initialize the feature extractor
    extractor = DeepFeatureExtractor()
    
    # Example: Extract features from a single image
    import glob
    import json

    # Step 1: Define the directory containing your images
    phish_directory = "mini_phishing/train_phishing" 
    benign_directory = "mini_phishing/train_benign"

    # Step 2: Collect all image paths (assuming .png and .jpg)
    phishing_images = glob.glob(os.path.join(phish_directory, "*.png")) + \
                glob.glob(os.path.join(phish_directory, "*.jpg"))
                
    legitimate_images = glob.glob(os.path.join(benign_directory, "*.png")) + \
                glob.glob(os.path.join(benign_directory, "*.jpg"))
    
    # Combine into one list
    all_images = phishing_images + legitimate_images

    # Extract features
    print("Extracting features...")
    features_dict = extractor.extract_features_batch(all_images)

    # Create labels
    labels_dict = {}
    for img_path in phishing_images:
        if img_path in features_dict:
            labels_dict[img_path] = 1  # phishing label

    for img_path in legitimate_images:
        if img_path in features_dict:
            labels_dict[img_path] = 0  # legitimate label

    # Save features
    np.savez('features.npz', **features_dict)

    # Save labels
    with open("labels.json", "w") as f_lbl:
        json.dump(labels_dict, f_lbl)


    print(f"Saved {len(features_dict)} feature vectors and labels.")