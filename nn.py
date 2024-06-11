import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import os
import glob
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Command-line arguments
parser = argparse.ArgumentParser(description='Train a microtubule tracking model.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading.')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images.')
parser.add_argument('--annotation_file', type=str, required=True, help='File containing annotations.')

args = parser.parse_args()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class MicrotubuleDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.image_paths.sort()  # Ensure images are in sequential order
        self.annotations = self._load_annotations(annotation_file)
        
    def _load_annotations(self, annotation_file):
        data = []
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) == 5:
                        frame, coord_num, x, y, zero = parts
                        data.append([int(frame), int(coord_num), float(x), float(y), int(zero)])
        except FileNotFoundError:
            logging.error(f"Annotation file {annotation_file} not found.")
            raise
        
        columns = ['frame', 'coord_number', 'x', 'y', 'zero']
        return pd.DataFrame(data, columns=columns)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            logging.warning(f"Image {image_path} not found or unable to read.")
            return self.__getitem__((idx + 1) % len(self.image_paths))
        
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        
        frame_number = idx  # Assuming the frame number corresponds to the index
        coordinates = self.annotations[self.annotations['frame'] == frame_number][['x', 'y']].values.astype(np.float32)
        if coordinates.shape[0] == 0:
            logging.warning(f"No coordinates found for frame {frame_number}.")
            return self.__getitem__((idx + 1) % len(self.image_paths))
        
        return torch.tensor(image), torch.tensor(coordinates)

# Custom collate function to handle batches with varying sizes
def custom_collate_fn(batch):
    images, coords = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(coord.shape[0] for coord in coords)
    padded_coords = torch.zeros((len(coords), max_len, 2))
    for i, coord in enumerate(coords):
        padded_coords[i, :coord.shape[0], :] = coord
    return images, padded_coords

# Model
class MicrotubuleTrackingModel(nn.Module):
    def __init__(self):
        super(MicrotubuleTrackingModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.rnn = nn.LSTM(input_size=32*128*128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)  # Output layer to predict x, y coordinates
    
    def forward(self, x):
        batch_size, c, h, w = x.size()
        
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Use the output from the last time step
        return x

# Training function
def train(model, dataloader, criterion, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, coords in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, coords = images.to(device), coords.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coords[:, 0, :])  # Only compare the first coordinate for simplicity
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}')

# Testing function
def test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, coords in tqdm(dataloader, desc="Testing"):
            images, coords = images.to(device), coords.to(device)
            outputs = model(images)
            print('Predicted Coordinates:', outputs)
            print('Actual Coordinates:', coords[:, 0, :])

# Main
if __name__ == "__main__":
    try:
        dataset = MicrotubuleDataset(args.image_dir, args.annotation_file)
    except FileNotFoundError:
        print("Annotation file not found. Exiting.")
        exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    torch.cuda.empty_cache()  # Clear GPU memory before training

    model = MicrotubuleTrackingModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloader, criterion, optimizer, num_epochs=args.num_epochs)
    test(model, dataloader)



