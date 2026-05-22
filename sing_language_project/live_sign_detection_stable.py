import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

print("🚀 Loading trained model and label encoder...")

# ==== Load trained model and label encoder ====
try:
    with open('sign_language_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print("✅ Model and encoder loaded successfully!")
except FileNotFoundError:
    print("❌ Model or encoder not found! Run final_train_model.py first.")
    exit()

# ==== Initialize MediaPipe ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ==== Start webcam safely ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

time.sleep(2)  # allow camera to warm up
print("🎥 Webcam started. Show your hand gesture! Press ESC to exit.")

# ==== Main loop ====
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame. Exiting...")
        break

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

            if len(data) == 63:
                data = np.array(data).reshape(1, -1)
                pred = model.predict(data)
                label = encoder.inverse_transform(pred)[0]
                gesture = label

    # ==== Display result ====
    cv2.putText(frame, f'Gesture: {gesture}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition (Stable)", frame)

    # waitKey MUST be > 1 for refresh; 1ms works fine
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        print("👋 Exiting...")
        break

# ==== Cleanup ====
cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
