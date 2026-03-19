import random
import cv2
import time
import numpy as np
import base64
import os
import threading
import queue
import pyaudio
from webrtcvad import Vad
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import mediapipe as mp
import smtplib
from email.message import EmailMessage
from datetime import datetime

# --- Flask App Initialization ---
app = Flask(__name__)

# --- OpenCV and MediaPipe Models ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Audio Detection Configuration ---
audio_event_queue = queue.Queue()
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_FRAME_MS = 30
AUDIO_CHUNK_SIZE = int(AUDIO_RATE * AUDIO_FRAME_MS / 1000)
VAD_MODE = 1  # Set to 1 for less aggressive audio filtering (as requested)


def audio_monitoring_thread(audio_queue):
    p = pyaudio.PyAudio()
    stream = None
    try:
        if p.get_device_count() == 0:
            print("No audio input devices found. Audio monitoring disabled.")
            return

        stream = p.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                        frames_per_buffer=AUDIO_CHUNK_SIZE)
        vad = Vad(VAD_MODE)
        print("Audio monitoring thread started...")
        while True:
            data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(data, AUDIO_RATE)
            if is_speech:
                audio_queue.put("speech_detected")
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in audio monitoring thread: {e}")
    finally:
        if stream:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        p.terminate()
        print("Audio monitoring thread stopped.")


# --- Application Data ---
APP_NAME = "Smart Proctor"
TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "password"
student_data = {}
incident_logs = []
chat_logs = []
active_otps = {}


@app.route('/')
def landing_page():
    return render_template('index.html', app_name=APP_NAME)


@app.route('/student_register')
def student_register_page():
    return render_template('register.html', app_name=APP_NAME)


@app.route('/student_login')
def student_login_page(error=None):
    return render_template('login.html', app_name=APP_NAME, error=error)


@app.route('/register', methods=['POST'])
def register():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    email = request.form.get('email')
    roll_number = request.form.get('roll_number')
    college_name = request.form.get('college_name')
    password = request.form.get('password')
    phone_number = request.form.get('phone_number')

    if roll_number in student_data:
        return render_template('register.html', app_name=APP_NAME,
                               error="Roll number already registered. Please login.")

    student_data[roll_number] = {
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'roll_number': roll_number,
        'college_name': college_name,
        'password': password,
        'phone_number': phone_number
    }
    print(f"New Student Registered: Roll={roll_number}")
    return redirect(url_for('student_login_page', error="Registration successful! Please log in."))


@app.route('/login', methods=['POST'])
def login():
    roll_number = request.form.get('roll_number')
    password = request.form.get('password')

    if roll_number not in student_data or student_data[roll_number]['password'] != password:
        return render_template('login.html', app_name=APP_NAME, error="Invalid roll number or password.")

    otp = str(random.randint(100000, 999999))
    active_otps[roll_number] = otp

    # --- SIMULATED OTP SENDING ---
    print("--- FAKE SMS SENDING ---")
    print(f"A simulated OTP has been generated for Roll Number {roll_number}.")
    print(f"The OTP is: {otp}")
    print(f"This OTP would be sent to phone number {student_data[roll_number]['phone_number']}.")
    print("--- FAKE SMS END ---")

    return render_template('otp_verification.html', app_name=APP_NAME, roll_number=roll_number)


@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    roll_number = request.form.get('roll_number')
    user_otp = request.form.get('otp')

    if roll_number in active_otps and active_otps[roll_number] == user_otp:
        del active_otps[roll_number]
        return redirect(url_for('register_face_page'))
    else:
        return render_template('otp_verification.html', app_name=APP_NAME, roll_number=roll_number,
                               error="Invalid OTP. Please try again.")


@app.route('/register_face_page')
def register_face_page():
    return render_template('register_face.html', app_name=APP_NAME)


@app.route('/search_colleges', methods=['GET'])
def search_colleges():
    search_query = request.args.get('q', '').lower()
    all_colleges = [
        "MIT University, Pune",
        "Bharti Vidyapeeth College of Engineering, Pune",
        "College of Engineering, Pune (COEP)",
        "Vishwakarma Institute of Technology, Pune (VIT)",
        "Symbiosis Institute of Technology, Pune",
        "Modern Education Society's College of Engineering, Pune"
    ]
    suggestions = [college for college in all_colleges if search_query in college.lower()]
    return jsonify(suggestions)


def send_otp_email(to_email, otp):
    print(
        f"OTP {otp} for Roll Number {request.form.get('roll_number')} has been generated. (Email sending is currently disabled in the code)")


@app.route('/register-face', methods=['POST'])
def register_face():
    capture_success, message = capture_face_data()
    if capture_success:
        return render_template('exam.html', app_name=APP_NAME)
    else:
        return jsonify({"error": f"Face registration failed: {message}"}), 400


def capture_face_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error (capture_face_data): Could not open webcam.")
        return False, "Webcam could not be opened. Check if in use or drivers."
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None or frame.size == 0:
        print("Error (capture_face_data): Could not grab frame or received empty frame from webcam.")
        return False, "Failed to capture image from webcam."
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        cv2.imwrite('face_image_registered.jpg', frame)
        print("Webcam frame captured and face detected successfully for final registration.")
        return True, "Face registered successfully."
    else:
        print("Error (capture_face_data): No face detected in the captured frame for registration.")
        return False, "No face detected during registration capture. Please position your face clearly."


@app.route('/detect-face-realtime', methods=['POST'])
def detect_face_realtime():
    data = request.get_json()
    if not data or 'image' not in data:
        print("Error (detect_face_realtime): No image data received.")
        return jsonify({"face_detected": False}), 400
    image_data = data['image']
    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            print("Error (detect_face_realtime): Could not decode image or received empty frame.")
            return jsonify({"face_detected": False}), 400
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_detected = len(faces) > 0
        return jsonify({"face_detected": face_detected})
    except Exception as e:
        print(f"Error (detect_face_realtime): Exception during frame processing: {e}")
        return jsonify({"face_detected": False}), 500


@app.route('/detect-movement', methods=['POST'])
def detect_movement():
    data = request.get_json()
    if not data or 'image' not in data:
        print("Error (detect_movement): No image data received from exam page.")
        return jsonify({"warning_type": "none", "warning_message": ""}), 400
    image_data = data['image']
    warning_type = "none"
    warning_message = ""
    frame_for_screenshot = None
    try:
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_for_screenshot = frame.copy()
        if frame is None or frame.size == 0:
            print("Error (detect_movement - visual): Could not decode image or received empty frame.")
            warning_type = "webcam_issue"
            warning_message = "Webcam stream issue! Please ensure camera is working."
        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                results = holistic.process(image_rgb)
                if not results.face_landmarks:
                    warning_type = "face_not_visible"
                    warning_message = "Please keep your face clearly visible and look at the screen!"
                if results.face_landmarks:
                    left_eye = results.face_landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYE]
                    right_eye = results.face_landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYE]
                    if left_eye.x > 0.55 or right_eye.x < 0.45:
                        warning_type = "gaze_away"
                        warning_message = "Suspicious gaze detected! Please look at the screen."

            while not audio_event_queue.empty():
                event = audio_event_queue.get()
                if event == "speech_detected":
                    warning_type = "speech_detected"
                    warning_message = "Unauthorized speech detected! Please remain silent."
                    break

    except Exception as e:
        print(f"Runtime Error in AI processing: {e}")
        warning_type = "visual_processing_error"
        warning_message = "Visual monitoring error. Check console."

    if warning_type != "none":
        print(f"Warning detected: Type={warning_type}, Message={warning_message}")
        if frame_for_screenshot is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            screenshot_filename = f"incident_{timestamp}_{warning_type}.jpg"
            text_position = (min(50, frame_for_screenshot.shape[1] - 300), min(50, frame_for_screenshot.shape[0] - 50))
            cv2.putText(frame_for_screenshot, warning_message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            if not os.path.exists('incidents'):
                os.makedirs('incidents')
            cv2.imwrite(os.path.join('incidents', screenshot_filename), frame_for_screenshot)
            print(f"Incident recorded: {screenshot_filename} for {warning_message}")
            incident_logs.append({
                "id": len(incident_logs) + 1,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "student_id": "student_123",
                "warning_type": warning_type,
                "warning_message": warning_message,
                "screenshot_path": os.path.join('incidents', screenshot_filename)
            })
        return jsonify({"warning_type": warning_type, "warning_message": warning_message}), 400
    else:
        return jsonify({"warning_type": "none", "warning_message": ""}), 200


@app.route('/exam-submitted')
def exam_submitted():
    return render_template('exam-submitted.html')


@app.route('/teacher_login')
def teacher_login_page():
    return render_template('teacher_login.html')


@app.route('/teacher_authenticate', methods=['POST'])
def teacher_authenticate():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
        sorted_incidents = sorted(incident_logs, key=lambda x: x['timestamp'], reverse=True)
        return render_template('teacher_dashboard.html', incidents=sorted_incidents)
    else:
        return render_template('teacher_login.html', error="Invalid credentials. Please try again.")


@app.route('/teacher_dashboard')
def teacher_dashboard():
    sorted_incidents = sorted(incident_logs, key=lambda x: x['timestamp'], reverse=True)
    return render_template('teacher_dashboard.html', incidents=sorted_incidents)


@app.route('/get_incident_screenshot/<path:filename>')
def get_incident_screenshot(filename):
    incident_dir = 'incidents'
    if not os.path.exists(incident_dir):
        os.makedirs(incident_dir)
    if os.path.exists(os.path.join(incident_dir, filename)):
        return send_from_directory(incident_dir, filename)
    else:
        return "File not found", 404


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_type = data.get('user_type')
    message = data.get('message')
    if user_type and message:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        chat_logs.append({"user": user_type, "timestamp": timestamp, "message": message})
        print(f"New chat message from {user_type}: {message}")
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid message data"}), 400


@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(chat_logs)


if __name__ == '__main__':
    audio_thread = threading.Thread(target=audio_monitoring_thread, args=(audio_event_queue,), daemon=True)
    audio_thread.start()
    if not os.path.exists('incidents'):
        os.makedirs('incidents')
    app.run(debug=True)
