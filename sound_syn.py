### Boot the SuperCollier server first before running sound_synth()!!!
from pythonosc import udp_client
import time

def sound_synth(midinote):
    # Setup the OSC client
    client = udp_client.SimpleUDPClient("127.0.0.1", 57120)
    
    # Send each frequency value with a delay
    
    print(f"Sending Midinote Number: {midinote} ")
    client.send_message("/from_python", midinote)


