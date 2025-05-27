import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import pyautogui
import sys
import time

# Load model and label encoder
model = tf.keras.models.load_model("gesture_control_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Screen size for cursor movement
screen_width, screen_height = pyautogui.size()

# Timing control to debounce actions
last_action_time = {
    "LeftClk": 0,
    "RightClk": 0,
    "ScreenShort": 0
}
action_delay = 1.0  # seconds

# Gesture control toggle
gesture_control_enabled = True

# Extract 21 (x, y) points
def extract_hand_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

# Webcam start
cap = cv2.VideoCapture(0)
print("📷 Starting real-time gesture recognition and cursor control...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_hand_landmarks(hand_landmarks)
            if len(features) == 42:
                input_tensor = np.array([features])
                prediction = model.predict(input_tensor, verbose=0)
                class_id = np.argmax(prediction)
                gesture_name = label_encoder.inverse_transform([class_id])[0]

                # GUI overlay for gesture name
                color = (0, 255, 0) if gesture_control_enabled else (0, 0, 255)
                status = "ON" if gesture_control_enabled else "OFF"
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Control: {status}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if gesture_control_enabled:
                    # Execute based on gesture
                    if gesture_name == "Move":
                        index_finger_tip = hand_landmarks.landmark[8]
                        cursor_x = int(index_finger_tip.x * screen_width)
                        cursor_y = int(index_finger_tip.y * screen_height)
                        pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                    elif gesture_name == "LeftClk" and current_time - last_action_time["LeftClk"] > action_delay:
                        pyautogui.click()
                        last_action_time["LeftClk"] = current_time

                    elif gesture_name == "RightClk" and current_time - last_action_time["RightClk"] > action_delay:
                        pyautogui.rightClick()
                        last_action_time["RightClk"] = current_time

                    elif gesture_name == "ScreenShort" and current_time - last_action_time["ScreenShort"] > action_delay:
                        pyautogui.screenshot("screenshot.png")
                        print("📸 Screenshot saved!")
                        last_action_time["ScreenShort"] = current_time

                    elif gesture_name == "Exit":
                        print("👋 Exit gesture detected. Closing...")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit()

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        gesture_control_enabled = not gesture_control_enabled
        print(f"🔁 Gesture control toggled {'ON' if gesture_control_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("✅ Exited successfully.")
