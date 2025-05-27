import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# Load the model and label encoder
model = tf.keras.models.load_model("gesture_control_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Function to extract 21 (x, y) points into flat list of 42 features
def extract_hand_landmarks(hand_landmarks):
    landmark_points = []
    for lm in hand_landmarks.landmark:
        landmark_points.extend([lm.x, lm.y])
    return landmark_points

# Start video capture
cap = cv2.VideoCapture(0)
print("📷 Starting real-time gesture recognition...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to act as a mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features
            features = extract_hand_landmarks(hand_landmarks)
            if len(features) == 42:
                input_tensor = np.array([features])  # Shape (1, 42)

                # Predict
                prediction = model.predict(input_tensor, verbose=0)
                class_id = np.argmax(prediction)
                gesture_name = label_encoder.inverse_transform([class_id])[0]

                # Show gesture name
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Exited successfully.")
