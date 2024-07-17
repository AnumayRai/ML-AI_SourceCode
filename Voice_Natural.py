from pydub import AudioSegment
import os

input_folder = "path/to/input/folder"
output_folder = "path/to/output/folder"
sample_rate = 16000

for file in os.listdir(input_folder):
    audio = AudioSegment.from_file(os.path.join(input_folder, file))
    audio = audio.set_frame_rate(sample_rate)
    audio.export(os.path.join(output_folder, file), format="wav")


import numpy as np
import librosa
import torch
from models import CycleGAN, WaveNet

class RealTimeProcessor:
    def __init__(self, cyclegan_path, wavenet_path):
        self.cyclegan = CycleGAN.load_from_checkpoint(cyclegan_path)
        self.wavenet = WaveNet.load_from_checkpoint(wavenet_path)

    def process(self, input_audio):
        # Convert input audio to Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(input_audio, sr=sample_rate)

        # Normalize the Mel-spectrogram
        mel_spectrogram = (mel_spectrogram - mel_mean) / mel_std

        # Convert Mel-spectrogram to tensor
        mel_spectrogram = torch.from_numpy(mel_spectrogram).unsqueeze(0).unsqueeze(0)

        # Perform voice conversion using CycleGAN
        converted_mel_spectrogram = self.cyclegan(mel_spectrogram)

        # Generate natural-sounding audio using WaveNet
        generated_audio = self.wavenet(converted_mel_spectrogram)

        return generated_audio.squeeze().cpu().numpy()
