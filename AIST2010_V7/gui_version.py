import pygame
import cv2
import mediapipe as mp
import os
import joblib
import numpy as np
import warnings
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import wavio as wv
import time
from sound_syn import sound_synth
from global_variable import *

# Initialize PyGame
pygame.init()

window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Hand Gesture Sound Synthesizer")
font = pygame.font.Font(None, 36)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_dir, "gesture_model.pkl")
model = joblib.load(model_file_path)
warnings.simplefilter("ignore")

def display_text_pygame(screen, text, position, color):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)


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


def create_button(screen, text, position, size, color):
    pygame.draw.rect(screen, color, (*position, *size))
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(
        center=(position[0] + size[0] // 2, position[1] + size[1] // 2)
    )
    screen.blit(text_surface, text_rect)
    return pygame.Rect(*position, *size)


def display_text_pygame(window, text, position, color, font_name="Arial", font_size=30):
    font = pygame.font.SysFont(font_name, font_size)
    text_surface = font.render(text, True, color)
    window.blit(text_surface, position)


def gui_mode(mode=0, midinote=69):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    init = False
    ft_mode_two = True
    finger_touched = False
    instrument_id = 0
    recording = False
    recording_start_time = None
    recording_file_path = None
    gesture_text = "No Gesture"  # Initialize gesture_text with a default value
    freq = 0  # Initialize freq with a default value
    recordings = 0
    # Flag to track if Fist gesture has been processed
    fist_processing = False

    # Get the default input device's information
    input_device_info = sd.query_devices(kind="input")
    input_channels = input_device_info["max_input_channels"]

    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if mode_button.collidepoint(mouse_pos):
                        mode = (mode + 1) % len(mode_id)
                        init = False
                        ft_mode_two = True
                        midinote = 69
                        recordings = 0
                    elif instrument_button.collidepoint(mouse_pos):
                        instrument_id = (instrument_id + 1) % len(instrument)
                    elif quit_button.collidepoint(mouse_pos):
                        print("Exiting the program...")
                        running = False

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Exiting...")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

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

                if mode == 0:
                    freq = np.clip((1 - middle_mcp.y) * 1980 + 20, 20, 2000)
                    sound_synth(
                        0 if gesture_text == "Fist" else freq, mode, instrument_id
                    )
                elif mode == 1:
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

                    index_finger_tip = results.multi_hand_landmarks[0].landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]

                    index_finger_tip_coords = (
                        int(index_finger_tip.x * frame.shape[1]),
                        int(index_finger_tip.y * frame.shape[0]),
                    )

                    thumb_middle_distance = calculate_distance(
                        thumb_tip_coords, 0, middle_finger_tip_coords
                    )

                    thumb_index_distance = calculate_distance(
                        thumb_tip_coords, index_finger_tip_coords, 0
                    )

                    if thumb_middle_distance < 40 or (
                        thumb_index_distance < 40 and ft_mode_two
                    ):  # If the distance between thumb and middle finger is less than 40 pixels
                        mcp_y = middle_mcp.y
                        midinote = 69
                        print("Position reset")
                        ft_mode_two = False
                        init = True
                    if init:
                        change_of_y = -(middle_mcp.y - mcp_y)
                        change_of_midi = int(change_of_y / midi_interval)
                        if abs(change_of_midi) > 0:
                            midinote += change_of_midi
                            mcp_y = middle_mcp.y
                    if gesture_text == "Fist":
                        sound_synth(0, mode, instrument_id)
                    else:
                        if (
                            thumb_index_distance < 40
                        ):  # If the distance between thumb and index finger is less than 40 pixels
                            if not finger_touched:
                                sound_synth(midinote, mode, instrument_id)
                                finger_touched = True
                        else:
                            sound_synth(0, mode, instrument_id)
                            finger_touched = False
                elif mode == 2:
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
                    elif recording:
                        recording = False
                        sd.stop()
                        recording_duration = time.time() - recording_start_time
                        print(f"Recording stopped: {recording_file_path}")
                        recordings += 1

                        wv.write(
                            recording_file_path,
                            recording_data[: int((recording_duration - 0.5) * 44100)],
                            44100,
                            sampwidth=2,
                        )
                        time.sleep(0.5)
                        client.send_message("/playfile", recording_file_path)

                    elif gesture_text == "Fist" and not fist_processing:
                        sound_synth(0, mode, instrument_id)
                        recordings -= 1 if recordings > 0 else 0
                        fist_processing = True

                    elif gesture_text != "Fist":
                        fist_processing = False

            else:  # No hands detected
                gesture_text = "No Hands"
                sound_synth(0, mode, instrument_id)
                fist_processing = False

            # Convert the frame to RGB (PyGame uses RGB, OpenCV uses BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to a smaller window
            small_frame = cv2.resize(rgb_frame, (480, 360))
            small_frame = np.rot90(small_frame)
            small_frame = pygame.surfarray.make_surface(small_frame)

            # Clear the screen
            window.fill((255, 255, 255))

            # Display the title at the top
            display_text_pygame(
                window,
                "Hand Gesture Detection",
                (window_width // 2 - 225, 25),
                (0, 0, 0),
                font_name="Comic Sans MS",  # Specify the font name
                font_size=40,  # Specify the font size
            )

            # Display the video frame on the left-hand side
            window.blit(small_frame, (60, 100))

            button_x = window_width - 400

            # Create buttons
            mode_button = create_button(
                window,
                "Mode",
                (button_x, 100),
                (200, 50),
                (0, 128, 255),
            )
            instrument_button = create_button(
                window,
                "Instrument",
                (button_x, 160),
                (200, 50),
                (0, 128, 255),
            )
            quit_button = create_button(
                window, "Quit", (button_x, 220), (200, 50), (255, 0, 0)
            )

            # Display texts on the right-hand side
            display_text_pygame(
                window,
                f"Mode: {mode_id[mode]}",
                (button_x, 300),
                (0, 0, 0),
            )

            display_text_pygame(
                window,
                f"Gesture: {gesture_text}",
                (button_x, 380),
                (0, 0, 0),
            )

            if mode == 0:
                display_text_pygame(
                    window,
                    f"Instrument: {instrument[instrument_id]}",
                    (button_x, 340),
                    (0, 0, 0),
                )
                display_text_pygame(
                    window,
                    f"Frequency: {0 if gesture_text == 'Fist' or gesture_text == 'No Hands' else freq:.0f}Hz",
                    (button_x, 420),
                    (0, 0, 0),
                )

            elif mode == 1:
                display_text_pygame(
                    window,
                    f"Instrument: {instrument[instrument_id]}",
                    (button_x, 340),
                    (0, 0, 0),
                )
                display_text_pygame(
                    window,
                    f"MIDI Note Number: {midinote}",
                    (button_x, 420),
                    (0, 0, 0),
                )

            elif mode == 2:
                display_text_pygame(
                    window,
                    f"Recordings: {recordings}",
                    (button_x, 340),
                    (0, 0, 0),
                )

            # Update the display
            pygame.display.flip()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

gui_mode(0, 69)
