import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
from sound_syn import sound_synth
from global_variable import *
import keyboard
#import time
import warnings

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, 'gesture_model.pkl')
model = joblib.load(model_file_path)
gestures = ["open_hand", "fist"]
warnings.simplefilter("ignore")

def detect_hand_gesture(mode = 0, midinote = 81): 
    # Initialize MediaPipe Hand detector
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    init = False
    present_mode_on = 0
    key_pressed = False  # State to track if the space key is pressed
    instrument_id = 0

    # Start capturing from the webcam
    cap = cv2.VideoCapture(1)
    # Initialize the hand tracking model
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7, static_image_mode=False) as hands:
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

            #Display useful messages on cv2 window
            mode_text = f"Mode: {mode_id[mode]}"
            cv2.putText(frame, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0 , 0), 2)

            instrument_text = f"Instrument: {instrument[instrument_id]}"
            cv2.putText(frame, instrument_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 , 0), 2)

            '''Press key operations'''    
            #Press 'm' to change mode
            if cv2.waitKey(respond_time) & 0xFF == ord('m'):
                mode = (mode + 1) % len(mode_id)
                init = False

            #Press 'p' to get into/out the presentation mode :)
            if cv2.waitKey(respond_time) & 0xFF == ord('p'):
                present_mode_on = (present_mode_on + 1) % 2
                if present_mode_on == 0:
                    print('Off')
                else: 
                    print('On')
                    i = 0 #Initialize the index of music_midi to play    

            #Press 'i' to change the instrument
            if cv2.waitKey(respond_time) & 0xFF == ord('i'):
                instrument_id = (instrument_id + 1) % len(instrument)
            
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
                    
                    # Use Middle_FINGER_MCP to record change of y-coordinate of hands
                    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    # ***The coordinates are reverse along x-axis(I don't know why...)

                    ''' ↓ For debug use
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    #mcp_coordinates_text = f"Middle MCP Y: {middle_mcp.y:.4f}"
                    #coordinates_text = f"Thumb Tip Y: {thumb_tip.y:.4f} | Index Tip Y: {index_tip.y:.4f} | Middle Tip Y: {middle_tip.y:.4f}"
                    # Display the gesture name on the frame
                    #cv2.putText(frame, gesture_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # Display the y-coordinates below the gesture text
                    #cv2.putText(frame, coordinates_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    #cv2.putText(frame, mcp_coordinates_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
                    '''
                    
                    if mode == 0: #Frequency mode
                        #Credit to Oscar's code
                        freq = np.clip(middle_mcp.y * 1980 + 20, 20, 2000)  # Map to desired frequency range
                        freq_text = f"Frequency: {freq:.0f}Hz"
                        cv2.putText(frame, freq_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if gesture_text == 'fist':
                                sound_synth(0, mode, instrument_id) #Stop sound synth when detecting fist
                        else:
                            sound_synth(freq, mode, instrument_id)
                    else:
                        midi_text = f"Midi Note Number: {midinote}"
                        cv2.putText(frame, midi_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        ###Perform Sound Synth

                        # Wait for the 's' key to capture capture the initial y position of the mcp
                        if keyboard.is_pressed('s'):
                            mcp_y = middle_mcp.y
                            print("Position reset")
                            init = True 

                        if init:
                            #Change the midinote according the change of mcp.y
                            # Time of the last processed key press
                            change_of_y = -(middle_mcp.y - mcp_y)
                            change_of_midi = int((change_of_y)/midi_interval)
                            if abs(change_of_midi) > 0:
                                midinote += change_of_midi
                                mcp_y = middle_mcp.y
                            if gesture_text == 'fist':
                                sound_synth(0, mode, instrument_id) #Stop sound synth when detecting fist
                            if keyboard.is_pressed('z'):
                                    if not key_pressed:
                                    # Check if the space bar is pressed and ensure a small delay
                                        if present_mode_on == 0:
                                            sound_synth(midinote, mode, instrument_id)
                                        else: 
                                            #Presentation Mode
                                            sound_synth(music_midi[i], mode, instrument_id)
                                            i += 1
                                        #time.sleep(0.5)
                                        key_pressed = True

                            # Reset the key_pressed state when the space bar is released
                            if not keyboard.is_pressed('z'):
                                key_pressed = False

            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
                
            # Wait for the 'q' key to be pressed to exitqqq
            if cv2.waitKey(respond_time) & 0xFF == ord('q'):
                print("Exiting the program...")
                break

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

detect_hand_gesture(0,81) #0: Frequency Mode 1: Discrete Midi Note Mode
