import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from sound_syn import sound_synth
from global_variable import *

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, 'gesture_model.pkl')
model = joblib.load(model_file_path)
gestures = ["open_hand", "fist"]

def detect_hand_gesture(mode = 0, midinote = 81): 
    # Initialize MediaPipe Hand detector
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    init = False
    current_midi = 0
    # Start capturing from the webcam
    cap = cv2.VideoCapture(1)
    # Initialize the hand tracking model
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
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

            # Wait for the 's' key to capture capture the initial y position of the wrist
            if cv2.waitKey(1) & 0xFF == ord('s'):
                wrist_y = wrist.y
                print("Position reset")
                init = True
            # Check if hand landmarks are detected

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    
                    landmarks = np.array(landmarks).reshape(1, -1)

                    # Predict the gesture
                    gesture_id = model.predict(landmarks)[0]
                    gesture_text = gestures[int(gesture_id)]
                    # Extract landmark positions for gesture recognition
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    ### We use wrist to record change of y-coordinate of our hands
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    # ***The coordinates are reverse along x-axis(I don't know why...)
                    wrist_coordinates_text = f"Wrist Y: {wrist.y:.4f}"
                    coordinates_text = f"Thumb Tip Y: {thumb_tip.y:.4f} | Index Tip Y: {index_tip.y:.4f} | Middle Tip Y: {middle_tip.y:.4f}"
                    if mode == 1:
                        # Display the gesture name on the frame
                        cv2.putText(frame, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        # Display the y-coordinates below the gesture text
                        cv2.putText(frame, coordinates_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, wrist_coordinates_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    midi_text = f"Midi Note Number: {midinote}"
                    cv2.putText(frame, midi_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    ###Perform Sound Synth
                    if init:
                        #Change the midinote according the change of wrist.y
                        change_of_y = -(wrist.y - wrist_y)
                        change_of_midi = int((change_of_y)/midi_interval)
                        if abs(change_of_midi) > 0:
                            midinote += change_of_midi
                            wrist_y = wrist.y
                        if gesture_text == 'fist':
                            sound_synth(0) #Stop sound synth when detecting fist
                        
                        #Avoid sending same midi note
                        if current_midi != midinote:
                            sound_synth(midinote)
                            current_midi = midinote

            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
                
            # Wait for the 'q' key to be pressed to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting the program...")
                break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

detect_hand_gesture(1,81) #0: Normal Mode 1:Debug Mode
