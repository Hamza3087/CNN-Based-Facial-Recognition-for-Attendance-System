import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load trained model
MODEL_PATH = "face_recognition_mobilenetv2.h5"
model = load_model(MODEL_PATH)

# Load Label Encoder
LABEL_ENCODER_PATH = "label_encoder.pkl"
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# MediaPipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# Target image size
IMG_SIZE = (224, 224)

def preprocess_image(image):
    """Preprocess image: detect face, crop, resize, and normalize."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_image)
    
    if not results.detections:
        return None
    
    face_data = results.detections[0].location_data.relative_bounding_box
    h, w, _ = image.shape
    x, y, w, h = int(face_data.xmin * w), int(face_data.ymin * h), int(face_data.width * w), int(face_data.height * h)
    x1, y1, x2, y2 = max(0, x), max(0, y), min(image.shape[1], x + w), min(image.shape[0], y + h)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    face_crop = image[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, IMG_SIZE)
    face_normalized = face_resized / 255.0
    
    return np.expand_dims(face_normalized, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to receive image, preprocess, and predict student roll number."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    processed_image = preprocess_image(image)
    if processed_image is None:
        return jsonify({"error": "No face detected"}), 400
    
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    roll_number = label_encoder.inverse_transform([predicted_class])[0]
    
    return jsonify({"roll_number": roll_number})

if __name__ == '__main__':
    app.run(debug=True)