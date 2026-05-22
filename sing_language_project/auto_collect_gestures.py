import cv2
import mediapipe as mp
import csv
import time
import os

# === Settings ===
OUTPUT_FILE = "hand_landmarks_ready.csv"  # master CSV file
SAMPLES_PER_GESTURE = 30                  # how many frames to record automatically
CAPTURE_DELAY = 0.4                       # seconds between frames

# === Setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# === Ask for gesture label ===
gesture_name = input("✋ Enter gesture name (e.g., hello, yes, no, done, wrong): ").strip().lower()
if gesture_name == "":
    print("⚠️ You must enter a gesture name! Exiting.")
    exit()

print(f"📸 Starting automatic capture for '{gesture_name}' gesture...")

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

# === Prepare CSV ===
file_exists = os.path.isfile(OUTPUT_FILE)
with open(OUTPUT_FILE, mode="a", newline="") as f:
    csv_writer = csv.writer(f)
    if not file_exists:
        # write header only once
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        csv_writer.writerow(header)

    saved = 0
    start_time = time.time()

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
                    print(f"✅ Saved sample {saved}/{SAMPLES_PER_GESTURE}")

                    # wait a bit between captures
                    time.sleep(CAPTURE_DELAY)

        cv2.putText(frame, f'Gesture: {gesture_name} ({saved}/{SAMPLES_PER_GESTURE})',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Auto Gesture Collector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit early
            print("👋 Exiting early...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Data collection for '{gesture_name}' completed and saved to {OUTPUT_FILE}")
