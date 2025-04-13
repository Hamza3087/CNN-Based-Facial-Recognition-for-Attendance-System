import os
import cv2
import numpy as np
import sqlite3
import datetime
from flask import Flask, render_template, request, redirect, url_for, session, Response, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import sqlite3


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Secure session management

# Load trained face recognition model
try:
    model = load_model("./model/face_recognition.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load Label Encoder for roll numbers
dataset_folder = "augmented_dataset"
if os.path.exists(dataset_folder):
    labels = list(set([filename.split('_')[0] for filename in os.listdir(dataset_folder)]))
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
else:
    print("Dataset folder not found!")
    labels = []
    label_encoder = None

# Database Connection
def get_db_connection():
    conn = sqlite3.connect("attendance.db", check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    roll_number TEXT, 
                    timestamp TEXT)''')
    return conn

# Dummy Admin Credentials
ADMIN_USER = "admin"
ADMIN_PASS = "password"

# Webcam Capture
def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Face Recognition & Attendance Marking
def recognize_face(image):
    if model is None or label_encoder is None:
        return None, "⚠️ System not properly initialized!"
    
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    roll_number = label_encoder.inverse_transform([predicted_class])[0]
    
    # Mark attendance
    conn = get_db_connection()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO attendance (roll_number, timestamp) VALUES (?, ?)", (roll_number, timestamp))
    conn.commit()
    conn.close()
    
    return roll_number, f"✅ Attendance marked for {roll_number} at {timestamp}"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["admin"] = True
            flash("✅ Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("❌ Invalid credentials!", "danger")
    return render_template("admin.html")

@app.route('/dashboard')
def dashboard():
    if "admin" not in session:
        flash("⚠️ You must be logged in to access the dashboard.", "warning")
        return redirect(url_for("admin"))
    
    conn = get_db_connection()
    cursor = conn.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
    records = cursor.fetchall()
    conn.close()
    
    return render_template("dashboard.html", records=records)

@app.route('/logout')
def logout():
    session.pop("admin", None)
    flash("👋 Logged out successfully.", "info")
    return redirect(url_for("home"))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        roll_number, message = recognize_face(frame)
        return message
    return "❌ Face not recognized! Try again."

if __name__ == '__main__':
    conn = sqlite3.connect("attendance.db")
    conn.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, roll_number TEXT, timestamp TEXT)")
    conn.close()
    print("✅ Database setup complete!")
    app.run(debug=True)
