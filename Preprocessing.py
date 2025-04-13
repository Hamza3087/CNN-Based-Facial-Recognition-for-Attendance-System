import os
import cv2
import numpy as np
import mediapipe as mp

# Create output directory if it doesn't exist
input_dir = "filtered_images"
output_dir = "preprocessed_images"
os.makedirs(output_dir, exist_ok=True)

# Set up MediaPipe face detector
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# Target image dimensions
TARGET_SIZE = (224, 224)

def normalize_img(img):
    # Scale pixel values to 0-255 range for proper saving
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Process each image in the input folder
for img_file in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_file)
    
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Couldn't read {img_file}, skipping...")
        continue
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    results = detector.process(rgb)
    
    if not results.detections:
        print(f"No face found in {img_file}")
        continue
    
    # Get the first face bounding box
    face_data = results.detections[0].location_data.relative_bounding_box
    height, width, _ = img.shape
    
    # Convert relative coordinates to absolute
    x = int(face_data.xmin * width)
    y = int(face_data.ymin * height)
    w = int(face_data.width * width)
    h = int(face_data.height * height)
    
    # Ensure box is within image boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    
    # Skip if empty crop area
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid face crop in {img_file}")
        continue
    
    # Crop out the face
    face_crop = img[y1:y2, x1:x2]
    
    # Resize to standard dimensions
    face_resized = cv2.resize(face_crop, TARGET_SIZE)
    
    # Normalize pixel values
    face_normalized = normalize_img(face_resized)
    
    # Save the processed image
    save_path = os.path.join(output_dir, img_file)
    cv2.imwrite(save_path, face_normalized)
    
    print(f"Processed {img_file}")

print("All images processed!")