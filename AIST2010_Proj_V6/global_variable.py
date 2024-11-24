from pythonosc import udp_client

midi_interval = 0.026 #Value of y to increase / decrease 1 midinote
respond_time = 1
mode_id = ['Continuous Pitch Mode', 'Discrete Pitch Mode', "Composing Mode"]
music_midi = [81, 79, 76, 79, 84, 81, 79, 81, 76, 79, 81, 79, 76, 74, 72, 79,76, 74, 74, 76, 79, 79, 81, 76, 74, 72, 79, 76, 74, 72, 69, 72, 67, 0]
instrument = ['Sine Oscillator', 'Hand Flute']
gestures = [
    "Open Hand",
    "Fist",
    "Cross",
    "One",
    "Two",
    "Three",
    "Thumb",
]
client = udp_client.SimpleUDPClient("127.0.0.1", 57120)

