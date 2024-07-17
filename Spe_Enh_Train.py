import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Initialize the model, loss function, and optimizer
model = Denoiser()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Load the dataset and create data loaders for training and validation
noisy_train_paths = [...]  # List of paths to noisy audio files for training
clean_train_paths = [...]  # List of paths to clean audio files for training
noisy_val_paths = [...]  # List of paths to noisy audio files for validation
clean_val_paths = [...]  # List of paths to clean audio files for validation
train_dataset = AudioDataset(noisy_train_paths, clean_train_paths)
val_dataset = AudioDataset(noisy_val_paths, clean_val_paths)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
num_epochs = 100
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for noisy_audio, clean_audio in train_dataloader:
        optimizer.zero_grad()
        output = model(noisy_audio)
        loss = criterion(output, clean_audio)
        loss.backward()
        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for noisy_audio, clean_audio in val_dataloader:
            output = model(noisy_audio)
            val_loss += criterion(output, clean_audio).item()
    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Learning Rate Scheduler and Early Stopping
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        print("Early stopping...")
        break
