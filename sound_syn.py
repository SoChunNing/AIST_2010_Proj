### Boot the SuperCollier server first before running sound_synth()!!!
from pythonosc import udp_client


def sound_synth(midi_freq, mode):
    # Setup the OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
    
    # Send each frequency value with a delay
    
    print(f"Sending Midinote Number: {midi_freq},  ")
    client.send_message("/from_python", [midi_freq, mode])






