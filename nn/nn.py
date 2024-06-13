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
import warnings
warnings.filterwarnings("ignore", message="Applied workaround for CuDNN issue, install nvrtc.so")

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Command-line arguments
parser = argparse.ArgumentParser(description='Train a microtubule tracking model.')
parser.add_argument('--train_image_dirs', nargs='+', required=True, help='List of directories containing training images.')
parser.add_argument('--test_image_dirs', nargs='+', default=None, help='List of directories containing testing images (optional).')
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
    def __init__(self, image_dirs):
        self.image_paths = []
        self.annotations = pd.DataFrame()
      
        for image_dir in image_dirs:
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))  # Ensure images are in sorted order
            self.image_paths.extend(image_paths)
          
            # Automatically find the annotation file in the image directory
            annotation_file = glob.glob(os.path.join(image_dir, '*.txt'))
            if len(annotation_file) != 1:
                raise FileNotFoundError(f"Expected one annotation file in directory {image_dir}, but found {len(annotation_file)}.")
            annotation_file = annotation_file[0]
            self.annotations = pd.concat([self.annotations, self._load_annotations(annotation_file)], ignore_index=True)
      
        # Skip the first frame
        self.image_paths = self.image_paths[1:]
        self.annotations = self.annotations[self.annotations['frame'] > 0]
      
    def _load_annotations(self, annotation_file):
        data = []
        try:
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        frame, coord_num, x, y, zero = parts
                        data.append([int(frame), int(coord_num), float(x), float(y), int(float(zero))])
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
    return images.unsqueeze(1), padded_coords  # Ensure images have an additional dimension for sequence length

# ConvLSTM layer implementation
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# Model
class MicrotubuleTrackingModel(nn.Module):
    def __init__(self, num_points=300):
        super(MicrotubuleTrackingModel, self).__init__()
        self.num_points = num_points
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
      
        self.convlstm = ConvLSTMCell(input_dim=16, hidden_dim=16, kernel_size=(3, 3), bias=True)
        self.fc = nn.Linear(16 * 128 * 128, 2 * self.num_points)  # Output layer to predict multiple x, y coordinates
  
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()  # include seq_len dimension

        x = x.view(-1, c, h, w)  # reshape to (batch_size * seq_len, c, h, w)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(batch_size, seq_len, 16, h // 4, w // 4)  # reshape back to (batch_size, seq_len, 16, h/4, w/4)

        # Initialize hidden state and cell state for ConvLSTM
        h, c = self.convlstm.init_hidden(batch_size, (x.size(3), x.size(4)))
        output_inner = []
        for t in range(seq_len):
            h, c = self.convlstm(x[:, t, :, :, :], [h, c])  # pass each timestep through ConvLSTM
            output_inner.append(h)

        x = torch.stack(output_inner, dim=1)  # (batch_size, seq_len, hidden_dim, h, w)
        x = x[:, -1, :, :, :]  # take the output of the last timestep
        x = x.view(batch_size, -1)  # flatten the output for the fully connected layer

        x = self.fc(x)
        return x.view(batch_size, self.num_points, 2)  # reshape to (batch_size, num_points, 2)

# Custom loss function using Hungarian algorithm for optimal matching
# Custom loss function using Hungarian algorithm for optimal matching
def custom_loss_function(predicted_coords, true_coords):
    batch_size, num_points, _ = predicted_coords.size()
    total_loss = 0
    weighted_term_weight = 0.1  # Adjust the weight of the weighted term as needed

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

        # Calculate the Euclidean distance for matched pairs (Hungarian loss)
        frame_loss += torch.norm(matched_pred - matched_true, dim=1).mean()

        # Weighted term to penalize the sum of squared distances between each predicted coordinate and the nearest true coordinate
        nearest_distances = torch.min(distance_matrix, dim=1)[0]  # Get the minimum distance for each predicted point
        weighted_term = torch.sum(nearest_distances**2)
        frame_loss += weighted_term_weight * weighted_term

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
        train_dataset = MicrotubuleDataset(train_image_dirs)
      
        if args.test_image_dirs:
            test_image_dirs = args.test_image_dirs
            test_dataset = MicrotubuleDataset(test_image_dirs)
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



