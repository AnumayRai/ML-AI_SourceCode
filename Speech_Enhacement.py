import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

# Define the CNN model
class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Define a custom dataset for loading the audio files
class AudioDataset(Dataset):
    def __init__(self, noisy_paths, clean_paths):
        self.noisy_paths = noisy_paths
        self.clean_paths = clean_paths

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_audio, _ = librosa.load(self.noisy_paths[idx], sr=16000, mono=True)
        clean_audio, _ = librosa.load(self.clean_paths[idx], sr=16000, mono=True)
        noisy_audio = torch.from_numpy(noisy_audio).float().unsqueeze(0)
        clean_audio = torch.from_numpy(clean_audio).float().unsqueeze(0)
        return noisy_audio, clean_audio

# Initialize the model, loss function, and optimizer
model = Denoiser()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the dataset and create a data loader
noisy_paths = [...]  # List of paths to noisy audio files
clean_paths = [...]  # List of paths to clean audio files
dataset = AudioDataset(noisy_paths, clean_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for noisy_audio, clean_audio in dataloader:
        optimizer.zero_grad()
        output = model(noisy_audio)
        loss = criterion(output, clean_audio)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
