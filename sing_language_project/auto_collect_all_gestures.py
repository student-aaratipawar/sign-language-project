import cv2
import mediapipe as mp
import csv
import time
import os

# === Settings ===
OUTPUT_FILE = "hand_landmarks_ready.csv"
SAMPLES_PER_GESTURE = 30
CAPTURE_DELAY = 0.4
PREP_TIME = 5  # seconds before each gesture recording starts

# === Gesture list ===
GESTURES = [
    "hello", "yes", "no", "done", "wrong",
    "stop", "peace", "love", "ok", "one",
    "two", "call", "bye", "wait", "thanks",
    "punch", "left", "right", "up", "down"
]

# === Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === Webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

# === Prepare CSV ===
file_exists = os.path.isfile(OUTPUT_FILE)
with open(OUTPUT_FILE, mode="a", newline="") as f:
    csv_writer = csv.writer(f)
    if not file_exists:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        csv_writer.writerow(header)

    for gesture_name in GESTURES:
        print(f"\n🖐️ Get ready to show gesture: '{gesture_name.upper()}'")
        for sec in range(PREP_TIME, 0, -1):
            print(f"⏳ Starting in {sec} seconds...", end="\r")
            time.sleep(1)

        print(f"📸 Recording '{gesture_name}' now!")
        saved = 0

        while saved < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not captured.")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data = []
                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    if len(data) == 63:
                        csv_writer.writerow(data + [gesture_name])
                        saved += 1
                        print(f"✅ {gesture_name}: Saved sample {saved}/{SAMPLES_PER_GESTURE}", end="\r")
                        time.sleep(CAPTURE_DELAY)

            cv2.putText(frame, f'{gesture_name} ({saved}/{SAMPLES_PER_GESTURE})',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Auto Gesture Collector", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                print("\n👋 Exiting early...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"\n✅ Completed gesture '{gesture_name}' ({SAMPLES_PER_GESTURE} samples).")
        time.sleep(2)

    print("\n🎉 All gestures collected successfully!")
    cap.release()
    cv2.destroyAllWindows()
