import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib

# Load trained model and label encoder
model = tf.keras.models.load_model("gesture_control_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam feed
cap = cv2.VideoCapture(0)

def extract_landmarks(results):
    """Extract normalized landmark positions from MediaPipe results"""
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = "No hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks and make prediction
        landmarks = extract_landmarks(results)
        if landmarks is not None and len(landmarks) == model.input_shape[1]:
            input_data = np.expand_dims(landmarks, axis=0)
            prediction_probs = model.predict(input_data)
            predicted_class = np.argmax(prediction_probs)
            prediction = label_encoder.inverse_transform([predicted_class])[0]

    # Display prediction
    cv2.putText(frame, f'Gesture: {prediction}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
