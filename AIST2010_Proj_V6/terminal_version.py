import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
import warnings
import time
import sounddevice as sd
import soundfile as sf
import wavio as wv
from scipy.io.wavfile import write
from sound_syn import sound_synth
from global_variable import *

# Initialize the OSC client
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = joblib.load(model_file_path)
warnings.simplefilter("ignore")


def display_text(frame, text, position, color):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def calculate_distance(thumb_tip, index_finger_tip, middle_finger_tip):
    if middle_finger_tip == 0:
        return np.sqrt(
            (thumb_tip[0] - index_finger_tip[0]) ** 2
            + (thumb_tip[1] - index_finger_tip[1]) ** 2
        )
    else:
        return np.sqrt(
            (thumb_tip[0] - middle_finger_tip[0]) ** 2
            + (thumb_tip[1] - middle_finger_tip[1]) ** 2
        )

def detect_hand_gesture(mode=0, midinote=81):
    init = False
    #Flags to check wether the keys is released
    key_pressed_s = key_pressed_z  = False
    instrument_id = 0
    present_mode_on = 0
    recording = False
    recording_start_time = None
    recording_file_path = None
    # Get the default input device's information
    input_device_info = sd.query_devices(kind="input")
    input_channels = input_device_info["max_input_channels"]

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            display_text(frame, f"Mode: {mode_id[mode]}", (10, 20), (255, 0, 0))
            display_text(
                frame,
                f"Instrument: {instrument[instrument_id]}",
                (10, 40),
                (255, 255, 0),
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("m"):
                    mode = (mode + 1) % len(mode_id)
                    print("Mode Changed")
                    init = False
            elif key == ord("i"):
                    instrument_id = (instrument_id + 1) % len(instrument)
                    print("Instrument Changed")
            elif key == ord("q"):
                print("Exiting the program...")
                break
            #Press 'p' to get into/out the presentation mode :)
            elif key == ord('p'):
                present_mode_on = (present_mode_on + 1) % 2
                if present_mode_on == 0:
                    print('Off')
                else: 
                    print('Ready')
                    i = 0 #Initialize the index of music_midi to play
                #Midi Note Number Fine-tuning
            elif key == ord('x'): 
                midinote += 1
            elif key == ord('z'): 
                midinote -= 1
           
            if results.multi_hand_landmarks:
                hand_landmarks_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    landmarks = np.array(
                        [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    ).flatten()
                    hand_landmarks_list.append(landmarks)

                if len(hand_landmarks_list) == 2:
                    # Concatenate landmarks of both hands
                    combined_landmarks = np.concatenate(hand_landmarks_list).reshape(
                        1, -1
                    )
                else:
                    # Pad the single hand landmarks to match the expected number of features
                    combined_landmarks = np.concatenate(
                        [hand_landmarks_list[0], np.zeros(42)]
                    ).reshape(1, -1)

                gesture_id = model.predict(combined_landmarks)[0]
                gesture_text = gestures[int(gesture_id)]
                middle_mcp = results.multi_hand_landmarks[0].landmark[
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                ]

                display_text(frame, f"Gesture: {gesture_text}", (10, 80), (0, 0, 255))

                match mode:
                    case 0:
                        freq = np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                        display_text(
                            frame, f"Frequency: {freq:.0f}Hz", (10, 60), (0, 255, 0)
                        )
                        sound_synth(
                            0 if gesture_text == "Fist" else freq, mode, instrument_id
                        )
                    case 1:
                        display_text(
                            frame, f"Midi Note Number: {midinote}", (10, 60), (0, 255, 0)
                        )

                        thumb_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.THUMB_TIP
                        ]
                        thumb_tip_coords = (
                            int(thumb_tip.x * frame.shape[1]),
                            int(thumb_tip.y * frame.shape[0]),
                        )
                        middle_finger_tip = results.multi_hand_landmarks[0].landmark[
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                        ]
                        middle_finger_tip_coords = (
                            int(middle_finger_tip.x * frame.shape[1]),
                            int(middle_finger_tip.y * frame.shape[0]),
                        )

                        thumb_middle_distance = calculate_distance(
                            thumb_tip_coords, 0, middle_finger_tip_coords
                        )

                        #Replace the 's' button
                        if thumb_middle_distance < 40:  # If the distance between thumb and middle finger is less than 40 pixels
                            if not key_pressed_s: 
                                mcp_y = middle_mcp.y
                                midinote = 69
                                print("Position Initialized")
                                init = True
                                key_pressed_s = True
                        else: key_pressed_s = False

                        if init:
                            change_of_y = -(middle_mcp.y - mcp_y)
                            change_of_midi = int(change_of_y / midi_interval)
                            if abs(change_of_midi) > 0:
                                midinote += change_of_midi
                                mcp_y = middle_mcp.y
                            match gesture_text:
                                case "Fist":
                                    sound_synth(0, mode, instrument_id)
                                case 'Open Hand':
                                    index_finger_tip = results.multi_hand_landmarks[0].landmark[
                                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                                    ]

                                    index_finger_tip_coords = (
                                        int(index_finger_tip.x * frame.shape[1]),
                                        int(index_finger_tip.y * frame.shape[0]),
                                    )

                                    thumb_index_distance = calculate_distance(
                                        thumb_tip_coords, index_finger_tip_coords, 0
                                    )
                                    #Relace the 'z' button
                                    if thumb_index_distance < 40:  # If the distance between thumb and index finger is less than 40 pixels
                                        if not key_pressed_z:
                                            if present_mode_on == 0:
                                                    sound_synth(midinote, mode, instrument_id)
                                            else: 
                                                #Presentation Mode
                                                sound_synth(music_midi[i], mode, instrument_id)
                                                i = (i + 1) % len(music_midi)
                                                #time.sleep(0.5)
                                            key_pressed_z = True
                                    else: key_pressed_z = False
                    case 2:
                        if gesture_text == "One":
                            if not recording:
                                recording = True
                                recording_start_time = time.time()
                                recording_file_path = os.path.join(
                                    script_dir,
                                    "recording",
                                    f"recording_{int(recording_start_time)}.wav",
                                )
                                print(f"Recording started: {recording_file_path}")
                                recording_data = sd.rec(
                                    int(
                                        10 * 44100
                                    ),  # Initial buffer size, will be adjusted later
                                    samplerate=44100,
                                    channels=1,
                                    dtype="float32",
                                )
                        else:
                            if recording:
                                recording = False
                                sd.stop()
                                recording_duration = time.time() - recording_start_time
                                print(f"Recording stopped: {recording_file_path}")

                                wv.write(
                                    recording_file_path,
                                    recording_data[
                                        : int((recording_duration - 0.5) * 44100)
                                    ],
                                    44100,
                                    sampwidth=2,
                                )
                                time.sleep(0.5)
                                client.send_message("/playfile", recording_file_path)

            cv2.imshow("Hand Gesture Recognition", frame)

    cap.release()
    cv2.destroyAllWindows()

detect_hand_gesture(0, 69)
