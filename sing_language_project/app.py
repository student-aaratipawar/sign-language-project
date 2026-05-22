from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import pickle
import subprocess
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and encoder if available
model_path = 'sign_language_model.pkl'
encoder_path = 'label_encoder.pkl'

model, encoder = None, None
if os.path.exists(model_path) and os.path.exists(encoder_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

# ================== ROUTES ==================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    subprocess.run(["python", "final_train_model.py"])
    return render_template('train.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load image
    image = cv2.imread(filepath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe to extract landmarks
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    result = hands.process(rgb)

    gesture = "No hand detected"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            if len(data) == 63 and model is not None:
                pred = model.predict(np.array(data).reshape(1, -1))
                label = encoder.inverse_transform(pred)[0]
                gesture = label

    return render_template('result.html', gesture=gesture, image_path=filepath)

# ======= Live Video Stream =======
def generate_frames():
    global model, encoder
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            gesture = "Detecting..."

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])
                    if len(data) == 63 and model is not None:
                        pred = model.predict(np.array(data).reshape(1, -1))
                        label = encoder.inverse_transform(pred)[0]
                        gesture = label

            cv2.putText(frame, f'Gesture: {gesture}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============= RUN APP =============
if __name__ == '__main__':
    app.run(debug=True)
