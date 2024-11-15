
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
		var freq;
		if (mode == 0) {
			freq = recive_num;
		} {
			freq = (440 * (2 ** ((recive_num - 69) / 12))); //Convert Midinote number to freq
		};
        // Print the received frequency value for debugging
        ("Changing frequency to: " + freq).postln;

        // Free (mute) the current Synth if it exists
        if (currentSynth.notNil) {
            currentSynth.free;  // Stops the current sound wave
        };
		if (recive_num == 0) {
			currentSynth.free;
		};
        // Create a new Synth with the updated frequency
        currentSynth = Synth(\simpleWave, [\freq, freq]);
        },
        '/from_python'  // OSC address to listen for
    );

    // Define a SynthDef for a simple sine wave oscillator
    SynthDef(\simpleWave, { |freq = 440, amp = 0.5|
        var sig;

        // Generate a sine wave oscillator with the specified frequency
        sig = SinOsc.ar(freq) * amp;

        // Output the signal to both left and right channels (stereo)
        Out.ar(0, sig ! 2);
    }).add;

s.boot();
