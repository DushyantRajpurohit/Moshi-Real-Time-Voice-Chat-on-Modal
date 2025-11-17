const { useRef, useEffect, useState } = React;

/**
 * Constructs the WebSocket URL for the Moshi backend service.
 * It modifies the current page's hostname to point to the Modal deployment's
 * Moshi service ('*-web' -> '*-moshi-web') and uses 'wss:' or 'ws:' protocol.
 *
 * @returns {string} The WebSocket URL.
 */
const getWebSocketURL = () => {
	const url = new URL(window.location.href);
	// Adjust the hostname to target the Moshi deployment's web endpoint
	const hostname = url.hostname.replace('-web', '-moshi-web');
	// Determine the correct WebSocket protocol (secure or insecure)
	const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';

	return `${protocol}//${hostname}/ws`;
}

/**
 * The main React component for the streaming audio-to-audio/text application.
 * It manages the WebSocket connection, audio recording, decoding, playback,
 * and text display.
 */
const App = () => {
	// Initialize AudioContext once with a fixed sample rate (48kHz)
	const [audioContext] = useState(() => new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 }));
	// State for the generated text output
	const [text, setText] = useState('');

	// Refs to maintain mutable values across renders
	const socketRef = useRef(null);
	const decoderRef = useRef(null);
	// Tracks the scheduled end time of the last played audio chunk for seamless playback
	const scheduledEndRef = useRef(null);

	/**
	 * Initiates the audio recording process using the Opus Recorder library.
	 * It requests microphone access and configures the recorder to stream Opus-encoded data.
	 */
	const startRecording = async () => {
		// Request microphone access
		const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

		// Configure the Opus recorder
		const rec = new Recorder({
			encoderPath: "https://cdn.jsdelivr.net/npm/opus-recorder@latest/dist/encoderWorker.min.js",
			streamPages: true,
			encoderApplication: 2049, // 2049 is VOIP/low-delay
			encoderFrameSize: 80, // Frame size in ms (80ms)
			encoderSampleRate: 24000,
			maxFramesPerPage: 1, // Send one Opus packet per WebSocket frame
			numberOfChannels: 1,
		});

		// Callback for when Opus data is available (i.e., every frame)
		rec.ondataavailable = async (data) => {
			// Send the encoded Opus data over the WebSocket if the connection is open
			if (socketRef.current?.readyState === WebSocket.OPEN) {
				await socketRef.current.send(data);
			}
		};

		await rec.start();
	};

	/**
	 * useEffect hook for initializing the OggOpusDecoder worker.
	 * Runs once on component mount.
	 */
	useEffect(() => {
		const init = async () => {
			// Initialize the decoder worker
			const decoder = new window['ogg-opus-decoder'].OggOpusDecoder();
			await decoder.ready;

			decoderRef.current = decoder;
		};
		init();
		// Cleanup function to free the decoder resources when the component unmounts
		return () => decoderRef.current?.free();
	}, []);

	/**
	 * Plays a chunk of audio data using the Web Audio API (AudioContext).
	 * It schedules playback to start immediately after the previously scheduled audio ends,
	 * ensuring seamless, gapless playback.
	 *
	 * @param {Float32Array} audioData The decoded PCM audio data.
	 */
	const playAudio = (audioData)=> {
		// Create an AudioBuffer
		const buffer = audioContext.createBuffer(1, audioData.length, audioContext.sampleRate);
		buffer.copyToChannel(audioData, 0);

		// Create a source node
		const source = audioContext.createBufferSource();
		source.buffer = buffer;
		source.connect(audioContext.destination);

		// Calculate the precise start time for gapless playback
		const startTime = Math.max(scheduledEndRef.current, audioContext.currentTime);
		source.start(startTime);
		// Update the scheduled end time for the next chunk
		scheduledEndRef.current = startTime + buffer.duration;
	};

	/**
	 * useEffect hook for establishing and managing the WebSocket connection.
	 * Runs once on component mount.
	 */
	useEffect(() => {
		const socket = new WebSocket(getWebSocketURL());
		socketRef.current = socket;

		// Once the socket opens, start recording audio
		socket.onopen = () => startRecording();

		/**
		 * Handles incoming messages from the Moshi service.
		 * Messages are prefixed with a tag byte: 1 for audio, 2 for text.
		 */
		socket.onmessage = async (event) => {
			const buffer = await event.data.arrayBuffer();
			// The first byte is the tag
			const tag = new Uint8Array(buffer)[0];
			// The rest is the payload (Opus bytes or Text bytes)
			const payload = buffer.slice(1);

			if (tag === 1) { // Audio data (Opus bytes)
				// Decode the incoming Opus packet using the worker
				const { channelData, samplesDecoded } = await decoderRef.current.decode(new Uint8Array(payload));
				if (samplesDecoded > 0) {
					// Play the decoded PCM audio
					playAudio(channelData[0]);
				}
			} else if (tag === 2) { // Text data
				// Decode the text payload
				const newText = new TextDecoder().decode(payload);
				// Append the new text to the state
				setText(prev => prev + newText);
			}
		};

		// Cleanup function to close the WebSocket connection on unmount
		return () => socket.close();
	}, []);

	// Render the main application UI
	return (
		<div className="bg-gray-900 text-white min-h-screen flex items-center justify-center p-4">
			<div className="bg-gray-800 rounded-lg shadow-lg w-full max-w-xl p-6">
				<p className="text-gray-300 break-words">
					{/* Display the generated text or a connection status message */}
					{text || 'Connecting...'}
				</p>
			</div>
		</div>
	);

};

// Render the main App component into the root element
ReactDOM.createRoot(document.getElementById("react")).render(<App />);