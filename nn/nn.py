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
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Command-line arguments
parser = argparse.ArgumentParser(description='Train a microtubule tracking model.')
parser.add_argument('--train_image_dirs', nargs='+', required=True, help='List of directories containing training images.')
parser.add_argument('--train_annotation_files', nargs='+', required=True, help='List of annotation files for training.')
parser.add_argument('--test_image_dirs', nargs='+', default=None, help='List of directories containing testing images (optional).')
parser.add_argument('--test_annotation_files', nargs='+', default=None, help='List of annotation files for testing (optional).')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading.')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 regularization).')

args = parser.parse_args()

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset
class MicrotubuleDataset(Dataset):
    def __init__(self, image_dirs, annotation_files):
        self.image_paths = []
        self.annotations = pd.DataFrame()
        
        for image_dir, annotation_file in zip(image_dirs, annotation_files):
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))  # Ensure images are in sorted order
            self.image_paths.extend(image_paths)
            self.annotations = pd.concat([self.annotations, self._load_annotations(annotation_file)], ignore_index=True)
        
        # Skip the first frame
        self.image_paths = self.image_paths[1:]
        self.annotations = self.annotations[self.annotations['frame'] > 0]
        
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
            # Skip the frame if image is not found or unable to read
            next_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(next_idx)
        
        image = cv2.resize(image, (512, 512))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        
        frame_number = idx + 1  # Adjust frame number to match skipping the first frame
        coordinates = self.annotations[self.annotations['frame'] == frame_number][['x', 'y']].values.astype(np.float32)
        if coordinates.shape[0] == 0:
            # Skip the frame if no coordinates are found
            next_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(next_idx)
        
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
    def __init__(self, num_points=300):
        super(MicrotubuleTrackingModel, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        
        self.rnn = nn.RNN(input_size=32*128*128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2 * self.num_points)  # Output layer to predict multiple x, y coordinates
    
    def forward(self, x):
        batch_size, c, h, w = x.size()
        
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Use the output from the last time step
        return x.view(batch_size, self.num_points, 2)  # Reshape to (batch_size, num_points, 2)

# Custom loss function using Hungarian algorithm for optimal matching
def custom_loss_function(predicted_coords, true_coords):
    batch_size, num_points, _ = predicted_coords.size()
    total_loss = 0

    for i in range(batch_size):
        pred = predicted_coords[i].to(device)
        true = true_coords[i].to(device)
        frame_loss = 0

        # Compute the pairwise distance matrix
        distance_matrix = torch.cdist(pred.unsqueeze(0), true.unsqueeze(0)).squeeze(0)

        # Use Hungarian algorithm to find the optimal matching
        row_ind, col_ind = linear_sum_assignment(distance_matrix.cpu().detach().numpy())
        matched_pred = pred[row_ind]
        matched_true = true[col_ind]

        # Calculate the Euclidean distance for matched pairs
        frame_loss += torch.norm(matched_pred - matched_true, dim=1).mean()
        
        total_loss += frame_loss
    
    return total_loss / batch_size

def train(model, dataloader, criterion, optimizer, num_epochs=20, view_last_epoch=False):
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    middle_frame_image = None
    middle_frame_pred_coords = None
    middle_frame_true_coords = None
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (images, coords) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, coords = images.to(device), coords.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # For mixed precision
                outputs = model(images)
                loss = criterion(outputs, coords)  # Custom loss function
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
            # Save the middle frame data during the last epoch
            if view_last_epoch and epoch == num_epochs - 1:
                if batch_idx == len(dataloader) // 2:
                    middle_frame_image = images[0].cpu().numpy().squeeze()
                    middle_frame_pred_coords = outputs[0].cpu().detach().numpy()
                    middle_frame_true_coords = coords[0].cpu().numpy()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader)}')
    
    if view_last_epoch and middle_frame_image is not None:
        # Plot the middle frame with predicted and true points
        plt.figure(figsize=(10, 10))
        plt.imshow(middle_frame_image, cmap='gray')
        plt.scatter(middle_frame_pred_coords[:, 0], middle_frame_pred_coords[:, 1], c='r', label='Predicted')
        plt.scatter(middle_frame_true_coords[:, 0], middle_frame_true_coords[:, 1], c='b', label='True')
        plt.legend()
        plt.title("Middle Frame: Predicted vs True Coordinates")
        plt.savefig("middle_frame_plot.png")  # Save the plot to a file
        print("Plot saved as middle_frame_plot.png")




# Testing function
def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, coords in tqdm(dataloader, desc="Testing"):
            images, coords = images.to(device), coords.to(device)
            outputs = model(images)
            loss = criterion(outputs, coords)  # Custom loss function
            test_loss += loss.item()
    print(f'Test Loss: {test_loss/len(dataloader)}')

# Main
if __name__ == "__main__":
    try:
        train_image_dirs = args.train_image_dirs
        train_annotation_files = args.train_annotation_files
        train_dataset = MicrotubuleDataset(train_image_dirs, train_annotation_files)
        
        if args.test_image_dirs and args.test_annotation_files:
            test_image_dirs = args.test_image_dirs
            test_annotation_files = args.test_annotation_files
            test_dataset = MicrotubuleDataset(test_image_dirs, test_annotation_files)
        else:
            test_dataset = None
    except FileNotFoundError as e:
        print(f"File not found: {e}. Exiting.")
        exit(1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn)

    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    else:
        test_dataloader = None

    torch.cuda.empty_cache()  # Clear GPU memory before training

    model = MicrotubuleTrackingModel().to(device)
    criterion = custom_loss_function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train(model, train_dataloader, criterion, optimizer, num_epochs=args.num_epochs, view_last_epoch=True)

    if test_dataloader:
        test(model, test_dataloader, criterion)



