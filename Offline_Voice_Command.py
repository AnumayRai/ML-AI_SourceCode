import sys
import wave
import pyaudio
import json
import vosk

# Set path to the language model
model_path = "path/to/model"

# Initialize the Vosk model
model = vosk.Model(model_path)

# Initialize PyAudio
p = pyaudio.PyAudio()

def listen_and_recognize():
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    rec = vosk.KaldiRecognizer(model, 16000)

    while True:
        data = stream.read(4000)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result["text"]:
                print(result["text"])
                # Process the recognized command here
        else:
            print("Failed to recognize speech")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    listen_and_recognize()

    
