import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead")

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from e2cnn import gspaces
from e2cnn import nn as e2nn

# Define the maximum number of coordinates
MAX_COORDS = 400

class MicrotubuleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.coordinates = []
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {self.data_dir}")
        subfolders = [f for f in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, f))]
        
        for subfolder in tqdm(subfolders, desc="Loading subfolders"):
            subfolder_path = os.path.join(self.data_dir, subfolder)
            # Load images
            images = []
            for image_file in sorted(os.listdir(subfolder_path)):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):
                    image_path = os.path.join(subfolder_path, image_file)
                    images.append(image_path)

            # Load coordinates from any .txt file
            txt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.txt')]
            if txt_files:
                coord_file = os.path.join(subfolder_path, txt_files[0])
                coords = []
                with open(coord_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            coords.append([float(parts[2]), float(parts[3])])
                
                self.image_paths.extend(sorted(images))
                self.coordinates.extend([coords] * len(images))
        
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Total coordinates sets loaded: {len(self.coordinates)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        coords = np.array(self.coordinates[idx])
        
        if self.transform:
            image = self.transform(image)

        # Ensure the number of coordinates matches MAX_COORDS
        if len(coords) > MAX_COORDS:
            coords = coords[:MAX_COORDS]
        elif len(coords) < MAX_COORDS:
            padding = np.zeros((MAX_COORDS - len(coords), 2))
            coords = np.vstack((coords, padding))

        return image, coords

class EquivariantVAE(nn.Module):
    def __init__(self):
        super(EquivariantVAE, self).__init__()
        
        # Define the rotation group
        r2_act = gspaces.Rot2dOnR2(N=8)
        
        # Encoder
        self.input_type = e2nn.FieldType(r2_act, 1*[r2_act.trivial_repr])
        self.conv1 = e2nn.R2Conv(self.input_type, e2nn.FieldType(r2_act, 32*[r2_act.regular_repr]), kernel_size=3, padding=1)
        self.relu1 = e2nn.ReLU(self.conv1.out_type)
        self.conv2 = e2nn.R2Conv(self.conv1.out_type, e2nn.FieldType(r2_act, 64*[r2_act.regular_repr]), kernel_size=3, padding=1)
        self.relu2 = e2nn.ReLU(self.conv2.out_type)
        self.conv3 = e2nn.R2Conv(self.conv2.out_type, e2nn.FieldType(r2_act, 128*[r2_act.regular_repr]), kernel_size=3, padding=1)
        self.relu3 = e2nn.ReLU(self.conv3.out_type)
        self.conv4 = e2nn.R2Conv(self.conv3.out_type, e2nn.FieldType(r2_act, 256*[r2_act.regular_repr]), kernel_size=3, padding=1)
        self.relu4 = e2nn.ReLU(self.conv4.out_type)
        
        # Fully connected layers
        self.fc_mu = nn.Linear(256 * 32 * 32, 512)
        self.fc_logvar = nn.Linear(256 * 32 * 32, 512)
        self.fc_decode = nn.Linear(512, 256 * 32 * 32)
        
        # Decoder
        self.convT1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.convT2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convT3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convT4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        # Coordinate prediction
        self.fc_coords = nn.Linear(512, MAX_COORDS * 2)  # Produces MAX_COORDS coordinate pairs

    def encode(self, x):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.tensor.view(x.tensor.size(0), -1)
        print("Shape before fully connected layers:", x.shape)  # Debugging line
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 256, 32, 32)
        z = self.sigmoid(self.convT4(self.convT3(self.convT2(self.convT1(z)))))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        coords = self.fc_coords(z)
        return recon_x, coords.view(coords.size(0), -1, 2), mu, logvar

def loss_function(recon_x, x, coords, target_coords, mu, logvar):
    BCE = nn.BCELoss(reduction='sum')(recon_x, x)
    MSE = nn.MSELoss(reduction='sum')(coords, target_coords)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + MSE + KLD

def train(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}") as pbar:
            for batch_idx, (data, target_coords) in enumerate(dataloader):
                data = data.float() / 255.0  # Normalize
                data = data.cuda() if torch.cuda.is_available() else data
                target_coords = target_coords.float()
                target_coords = target_coords.cuda() if torch.cuda.is_available() else target_coords
                optimizer.zero_grad()
                recon_batch, coords, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, coords, target_coords, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                pbar.update(1)

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
        
    # Save the model after training
    model_save_path = 'equivariant_vae_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    data_dir = 'C:/Users/Jackson/Documents/mt_data/preprocessed'  # Replace with your data directory
    batch_size = 16
    learning_rate = 1e-3
    epochs = 10
    num_workers = 4  # Number of worker processes for data loading

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MicrotubuleDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = EquivariantVAE().cuda() if torch.cuda.is_available() else EquivariantVAE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, epochs=epochs)

if __name__ == "__main__":
    main()
