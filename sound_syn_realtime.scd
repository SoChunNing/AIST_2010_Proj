// Declare the currentSynth variable in the main scope
var currentSynth;
currentSynth = nil;

// Define the OSC responder to handle frequency updates
OSCdef.new(
    \updateFrequency,
    { |msg|
        var freq = msg[1]; // Directly use the frequency value sent from Python
        ("Changing frequency to: " + freq).postln;

        // Free (mute) the current Synth if it exists
        if (currentSynth.notNil) {
            currentSynth.free; // Stops the current sound wave
            ("Synth freed").postln; // Debug output
        };

        // Create a new Synth with the updated frequency
        currentSynth = Synth(\simpleWave, [\freq, freq]);
        ("New synth created at frequency: " + freq).postln; // Debug output
    },
    '/from_python' // OSC address to listen for frequency updates
);

// Define a SynthDef for a simple sine wave oscillator
SynthDef(\simpleWave, { |freq = 440| // Only use frequency
    var sig;

    // Generate a sine wave oscillator with the specified frequency
    sig = SinOsc.ar(freq) * 0.5; // Use a fixed amplitude of 0.5 for testing

    // Output the signal to both left and right channels (stereo)
    Out.ar(0, sig ! 2);
}).add;

// Boot the server if it isn't already running
s.boot;