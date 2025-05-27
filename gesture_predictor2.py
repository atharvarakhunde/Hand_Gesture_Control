import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time
import webbrowser
from pynput.keyboard import Controller, Key
import pyautogui
import sys

# Open Slither.io in browser
webbrowser.open("https://slither.io")
time.sleep(5)  # Give time for browser to load

# Load model and label encoder
model = tf.keras.models.load_model("gesture_control_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Keyboard controller
keyboard = Controller()
boosting = False  # track boost state

# Screen resolution
screen_width, screen_height = pyautogui.size()

def extract_hand_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

def release_boost():
    global boosting
    if boosting:
        keyboard.release(Key.up)
        boosting = False

def apply_gesture(gesture_name, hand_landmarks):
    global boosting

    if gesture_name == "Move":
        release_boost()
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * screen_width)
        y = int(index_tip.y * screen_height)
        pyautogui.moveTo(x, y, duration=0.05)

    elif gesture_name == "ScreenShort":
        if not boosting:
            keyboard.press(Key.up)
            boosting = True

    elif gesture_name == "NotMove":
        release_boost()

# Start webcam
cap = cv2.VideoCapture(0)
print("🎮 Gesture Control Running... (Move, ScreenShort, NotMove)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = extract_hand_landmarks(hand_landmarks)
            if len(features) == 42:
                input_tensor = np.array([features])
                prediction = model.predict(input_tensor, verbose=0)
                class_id = np.argmax(prediction)
                gesture_name = label_encoder.inverse_transform([class_id])[0]

                # Display gesture name
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                apply_gesture(gesture_name, hand_landmarks)
    else:
        release_boost()

    cv2.imshow("Slither.io Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_boost()
cap.release()
cv2.destroyAllWindows()
print("✅ Exited successfully.")
