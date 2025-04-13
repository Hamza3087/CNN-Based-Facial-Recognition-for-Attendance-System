import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define constants
MODEL_PATH = "model/face_recognition_cnn.h5"
TEST_IMAGE_PATH = "i200507_2.jpg"  # Provide a single image file or a directory
IMG_SIZE = (224, 224)

# Load the trained model
model = load_model(MODEL_PATH)

# Load the label map used in training
label_map = {}  
label_map_path = "label_map.txt"  # Save label mapping in training

# Load the label map if available
if os.path.exists(label_map_path):
    with open(label_map_path, "r") as file:
        for line in file:
            roll_number, index = line.strip().split(":")
            label_map[int(index)] = roll_number  # Map index back to roll number
else:
    print("Warning: Label map file not found. Predictions will be numeric.")

# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to predict a single image
def predict_image(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions)

    roll_number = label_map.get(predicted_label, f"Unknown ({predicted_label})")
    print(f"Predicted Roll Number: {roll_number}")

# Check if the provided path is a directory or a file
if os.path.isdir(TEST_IMAGE_PATH):
    print("Processing multiple images in directory...")
    for filename in os.listdir(TEST_IMAGE_PATH):
        file_path = os.path.join(TEST_IMAGE_PATH, filename)
        predict_image(file_path)
else:
    print("Processing a single image...")
    predict_image(TEST_IMAGE_PATH)
