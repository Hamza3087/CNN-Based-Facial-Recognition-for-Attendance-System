import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Paths
dataset_folder = "augmented_dataset"

# Image parameters
IMG_SIZE = (224, 224)  # MobileNetV2 expects 224x224 images

# Load dataset
images = []
labels = []

for filename in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, filename)

    # Read and preprocess image
    image = cv2.imread(file_path)
    if image is None:
        continue

    image = cv2.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalize to [0,1]

    # Extract roll number as label
    roll_number = filename.split('_')[0]  # e.g., "i210834"

    images.append(image)
    labels.append(roll_number)

# Convert to NumPy arrays
images = np.array(images, dtype=np.float32)

labels = np.array(labels)

# Encode labels (roll numbers → unique integers)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
NUM_CLASSES = len(set(labels_encoded))  # Unique roll numbers
labels_one_hot = to_categorical(labels_encoded, NUM_CLASSES)

# Split dataset into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Load MobileNetV2 without the top layer
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the base model
base_model.trainable = False

# Define the new model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Reduce feature map size
    layers.Dense(256, activation="relu"),  # Fully connected layer
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(NUM_CLASSES, activation="softmax")  # Output layer
])

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"✅ Validation Accuracy: {val_acc:.4f}")

# Save model
model.save("face_recognition_mobilenetv2.h5")
print("✅ Model saved as face_recognition_mobilenetv2.h5")
