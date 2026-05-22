import cv2
import mediapipe as mp
import numpy as np
import pickle

print("🚀 Loading trained model...")

# Load the trained model
try:
    with open('sign_language_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ sign_language_model.pkl not found! Please run train_sign_model.py first.")
    exit()

# Try to load label encoder (for readable labels)
try:
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print("✅ Label encoder loaded successfully!")
except FileNotFoundError:
    encoder = None
    print("⚠️ label_encoder.pkl not found! Predictions will be numeric only.")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.startWindowThread()
cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Sign Language Recognition", 900, 700)

if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("🎥 Webcam started. Show your hand gesture! Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
