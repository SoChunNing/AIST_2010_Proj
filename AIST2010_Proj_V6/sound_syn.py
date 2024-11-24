### Boot the SuperCollier server first before running sound_synth()!!!
from global_variable import client

def sound_synth(midi_freq, mode, instrument_id):
    # Send each frequency value with a delay
    print(f"Sending Frequency/Midi Note Number: {midi_freq} ")
    client.send_message("/from_python", [midi_freq, mode, instrument_id])












