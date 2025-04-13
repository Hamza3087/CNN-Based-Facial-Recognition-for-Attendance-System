# CNN-Based-Facial-Recognition-for-Attendance-System

This project implements a Facial Recognition-based Attendance System using Convolutional Neural Networks (CNNs). The system captures student faces, recognizes them using deep learning, and automatically marks attendance. It integrates with a web-based user interface (UI) for easy access and use.

🚀 Objectives
Data Preprocessing: Load, preprocess, and augment face dataset.

CNN Model: Train a CNN-based face recognition model for attendance.

Web System: Build a UI for Admin login and Student attendance using webcam capture.

Model Evaluation: Assess model accuracy and other metrics like Precision, Recall, and F1-score.

Hyperparameter Tuning: Optimize model performance by tuning parameters like convolutional layers and learning rate.

🧩 Project Structure
main.py: Main script that:

Handles data preprocessing and augmentation.

Defines the CNN model for face recognition.

Integrates with the Flask web app for attendance marking.

app.py: The Flask web application that provides a UI for the system (Admin and Student modes).

requirements.txt: List of required Python packages.

templates/: Folder containing HTML templates for Flask web app.

static/: Folder containing static files (CSS, JS, images) for the UI.

dataset/: Folder for storing the preprocessed face images (uploaded by students).

models/: Folder for storing the trained CNN model.

🛠️ Requirements
Install dependencies using the following:
pip install -r requirements.txt

⚙️ CNN Model Development
CNN Architecture:

Convolutional layers to extract facial features.

MaxPooling layers for down-sampling.

Fully connected layers to classify faces.

Use ReLU for activation and Softmax for classification.

Model Training:

Train the CNN model using the preprocessed dataset.

Use an appropriate optimizer like Adam, and evaluate using loss and accuracy metrics.

Hyperparameter Tuning:

Tune parameters such as the number of layers, filters, learning rate, and batch size.

Use Grid Search or Random Search for optimization.

Model Evaluation:

Evaluate model accuracy on a validation/test set.

Use metrics such as:

Accuracy

Precision

Recall

F1-Score

Generate a Confusion Matrix.

💻 System Development and UI Integration
Admin Login:

A secure login page for authorized personnel to access and manage the system.

Student Attendance:

Webcam capture system to detect student faces.

Matches recognized faces with stored records, and automatically marks attendance with:

Student ID

Date and Time

Web Interface:

Built using Flask framework for easy interaction.

Real-time face recognition on webcam capture for student attendance.

📊 Visualization and Feature Analysis
Training/Validation Curves:

Plot the accuracy and loss curves for both training and validation data.

Confusion Matrix:

Visualize confusion matrix to analyze model performance.

Feature Maps:

Extract and visualize feature maps from the CNN to understand its decision-making process.

✅ Usage
Preprocess Dataset:

Store student face images in the dataset/ folder after uploading via the Google Form.

Train Model:

Run the CNN model training script to create a trained face recognition model.

Run Flask App:

Start the Flask app to access the admin login and student attendance system:

python 7-app.py

📈 Model Evaluation Metrics
Accuracy: The percentage of correct predictions.

Precision, Recall, and F1-Score: To assess the model's performance on imbalanced datasets.

Confusion Matrix: To analyze true positives, false positives, true negatives, and false negatives.

🔧 Hyperparameter Tuning
Number of Convolutional Layers and Filters: Experiment with different architectures.

Learning Rate and Batch Size: Fine-tune for optimal performance.

Optimizer: Try optimizers like Adam or SGD for training.

🖼️ Visualizations
Training/Validation Accuracy/Loss: Plot the training and validation accuracy/loss curves.

Confusion Matrix: Visualize a heatmap of the confusion matrix.

Feature Maps: Visualize activations in the convolutional layers to understand how the model learns.

🎓 Model Deployment
The system will automatically store attendance records in a database (SQLite or PostgreSQL) whenever a student is recognized.


