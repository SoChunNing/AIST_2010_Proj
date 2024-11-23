// Declare the currentSynth variable in the main scope
var currentSynth;
// Initialize currentSynth to nil
currentSynth = nil;

// Define the OSC responder to handle frequency updates
OSCdef.new(
    \updateFrequency,  // Unique identifier for the OSC listener
    { |msg|
        var recive_num = msg[1]; // Extract the frequency value from the OSC message
        var mode = msg[2];
        var instrument_id = msg[3];
        var freq;
        if (mode == 0) {
            freq = recive_num;
        } {
            freq = (440 * (2 ** ((recive_num - 69) / 12))); // Convert Midinote number to freq
        };
        // Print the received frequency value for debugging
        ("Changing frequency to: " + freq).postln;

        // Initialize the Synth if it doesn't exist
        if (currentSynth.isNil) {
            if (instrument_id == 0) {
                currentSynth = Synth(\sineWave, [\freq, freq]); // Sine Oscillator
            } {
                currentSynth = Synth(\handflute, [\freq, freq]); // Hand Flute
            };
        } {
            // Update the frequency of the existing Synth
            currentSynth.set(\freq, freq);
        };

        // Stop the Synth if the received frequency is 0
        if (recive_num == 0) {
            currentSynth.free;
            currentSynth = nil;
        };
    },
    '/from_python'  // OSC address to listen for
);

// Define a SynthDef for a simple sine wave oscillator
SynthDef(\sineWave, { |freq = 440, amp = 0.5|
    var sig;

    // Generate a sine wave oscillator with the specified frequency
    sig = SinOsc.ar(freq) * amp;

    // Output the signal to both left and right channels (stereo)
    Out.ar(0, sig ! 2);
}).add;

// Create a buffer for the audio file
~buffer = Buffer.read(s, "music_source/hand-flute-middle_filtered.wav");

SynthDef(\handflute, {
    |out = 0, freq = 880, originalFreq = 830.61, amp = 0.5|

    var playbackRate, signal;

    // Calculate the playback rate to change the frequency from originalFreq to targetFreq
    playbackRate = freq / originalFreq;

    // Play the recorded audio with the calculated playback rate
    signal = PlayBuf.ar(1, ~buffer.bufnum, rate: playbackRate, loop: 1);

    // Output the sound with volume control
    Out.ar(out, signal * amp ! 2);
}).add;

// Boot the server
s.boot();