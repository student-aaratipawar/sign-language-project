import csv
import os

# ✅ Ask for gesture name first (before importing TensorFlow)
gesture_name = input("✋ Enter gesture name (e.g., Hello, Yes, No): ").strip()

if gesture_name == "":
    print("⚠️ You must enter a gesture name! Please run again and type a label.")
    exit()

# 👇 Now import these after input()
import cv2
import mediapipe as mp


# Initialize Mediapipe AFTER input() to avoid blocking stdin
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# CSV file setup
csv_file = 'hand_landmarks_labeled.csv'
file_exists = os.path.isfile(csv_file)

# Open the CSV file for appending data
with open(csv_file, mode='a', newline='') as f:
    csv_writer = csv.writer(f)
    
    # Write header only once
    if not file_exists:
        header = []
        for i in range(21):
            header += [f'x{i}', f'y{i}', f'z{i}']
        header.append('label')
        csv_writer.writerow(header)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        exit()

    print("🎥 Collecting data... Press 's' to save a frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not captured.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Gesture: {gesture_name} | Press 's' to save, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                # ✅ FIXED: Add label name to the end of the row
                data.append(gesture_name)
                csv_writer.writerow(data)
                print(f"✅ Saved one sample for '{gesture_name}'")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"👋 Data collection for '{gesture_name}' finished.")
