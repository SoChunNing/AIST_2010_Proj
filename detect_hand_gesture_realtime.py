import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
import pandas as pd
import warnings
from pythonosc import udp_client
import time

warnings.filterwarnings("ignore", message="X does not have valid feature names")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = joblib.load(model_file_path)
gestures = [
    "palm",  # 01_palm
    "ok",  # 02_ok
    "fist",  # 03_fist
]


def calculate_frequency_based_on_hand_position(hand_landmarks, mp_hands):
    thumb_tip = hand_landmarks[mp_hands.HandLandmark.THUMB_TIP]
    frequency = np.clip(
        thumb_tip.y * 1980 + 20, 20, 2000
    )  # Map to desired frequency range
    return frequency


def detect_hand_gesture(mode=0):
    # Initialize MediaPipe Hand detector
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Start capturing from the webcam
    cap = cv2.VideoCapture(1)

    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

    # Initialize the hand tracking model
    with mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while True:  # Keep the loop running until the user decides to exit
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break

            # Flip the frame for a mirror view
            frame = cv2.flip(frame, 1)
            # Convert the frame color to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    landmarks = np.array(landmarks).reshape(1, -1)
                    print(f"Extracted landmarks: {landmarks}")

                    # Predict the gesture
                    gesture_id = model.predict(landmarks)[0]
                    gesture_text = gestures[int(gesture_id)]
                    print(f"Predicted gesture: {gesture_text}")

                    if gesture_text == "palm":
                        frequency = calculate_frequency_based_on_hand_position(
                            hand_landmarks.landmark, mp_hands
                        )
                        print(f"Gesture: {gesture_text}, Frequency: {frequency}")
                        client.send_message("/from_python", frequency)
                        # time.sleep(0.5)

                    # Extract landmark positions for gesture recognition
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    middle_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ]
                    # ***The coordinates are reverse along x-axis(I don't know why...)

                    coordinates_text = f"Thumb Tip Y: {thumb_tip.y:.4f} | Index Tip Y: {index_tip.y:.4f} | Middle Tip Y: {middle_tip.y:.4f}"
                    if mode == 1:
                        # Display the gesture name on the frame
                        cv2.putText(
                            frame,
                            gesture_text,
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                        # Display the y-coordinates below the gesture text
                        cv2.putText(
                            frame,
                            coordinates_text,
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

            # Show the frame
            cv2.imshow("Hand Gesture Recognition", frame)

            # Wait for the 'q' key to be pressed to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting the program...")
                break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()


detect_hand_gesture(1)  # 0: Normal Mode 1:Debug Mode
