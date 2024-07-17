import numpy as np
import pyaudio
import deepspeech

# DeepSpeech model configuration
model_path = "path/to/deepspeech-0.9.3-models.pbmm"
scorer_path = "path/to/deepspeech-0.9.3-models.scorer"
beam_width = 500
lm_alpha = 0.75
lm_beta = 1.85

# Audio configuration
sample_rate = 16000
chunk_size = int(sample_rate * 0.5)  # Process 0.5 seconds of audio at a time

# Initialize DeepSpeech model
model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)
model.setBeamWidth(beam_width)
model.setScorerAlphaBeta(lm_alpha, lm_beta)

# Initialize PortAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

# Buffer to store audio data
audio_buffer = np.zeros((chunk_size,))

# Transcription loop
print("Listening... Press Ctrl+C to stop.")
try:
    while True:
        # Capture audio from the microphone
        audio_data = stream.read(chunk_size)
        audio_buffer[:] = np.frombuffer(audio_data, dtype=np.int16)

        # Perform speech-to-text
        transcription = model.stt(audio_buffer)
        print(transcription, end="\r", flush=True)
except KeyboardInterrupt:
    print("\nStopping transcription.")

# Close PortAudio stream and release resources
stream.stop_stream()
stream.close()
p.terminate()
