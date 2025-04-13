import os
import cv2
import numpy as np
import albumentations as A

# Paths
input_folder = "preprocessed_images"
output_folder = "augmented_images"
os.makedirs(output_folder, exist_ok=True)

# Copy preprocessed images to augmented_images
for filename in os.listdir(input_folder):
    src_path = os.path.join(input_folder, filename)
    dest_path = os.path.join(output_folder, filename)
    cv2.imwrite(dest_path, cv2.imread(src_path))

# Define augmentation pipeline with more techniques
augmentation = A.Compose([
    A.Rotate(limit=30, p=0.7),  # Rotate by ±30 degrees
    A.HorizontalFlip(p=0.5),  # Horizontal Flip
    A.VerticalFlip(p=0.3),  # Vertical Flip
    A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast
    A.GaussianBlur(p=0.3),  # Apply Gaussian Blur
    A.RandomGamma(p=0.4),  # Adjust gamma values
    A.CLAHE(clip_limit=4.0, p=0.3),  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    A.ISONoise(p=0.3),  # Add ISO noise
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Apply elastic transformation
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Apply grid distortion
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3)  # Randomly drop parts of the image
])

# Track the highest number for each roll number
rollno_count = {}

# Initialize roll number counts from existing images in augmented_images
for filename in os.listdir(output_folder):
    parts = filename.rsplit("_", 1)
    if len(parts) == 2 and parts[1].split(".")[0].isdigit():
        roll_no = parts[0]
        num = int(parts[1].split(".")[0])
        rollno_count[roll_no] = max(rollno_count.get(roll_no, 0), num)

# Process images for augmentation
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Load image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Skipping invalid image: {filename}")
        continue
    
    # Extract roll number
    roll_no = "_".join(filename.split("_")[:-1])  # Extract roll number without count
    rollno_count[roll_no] = rollno_count.get(roll_no, 0)
    
    # Generate augmented images
    for i in range(30):
        rollno_count[roll_no] += 1
        new_filename = f"{roll_no}_{rollno_count[roll_no]}.jpg"
        output_path = os.path.join(output_folder, new_filename)
        
        augmented = augmentation(image=image)['image']
        cv2.imwrite(output_path, augmented)
        print(f"Saved augmented image: {new_filename}")

print("Augmentation complete.")