import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# Define the maximum number of coordinates
MAX_COORDS = 400

class MicrotubuleDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
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
        image = cv2.resize(image, (256, 256))  # Reduced image size
        coords = np.array(self.coordinates[idx])

        # Ensure the number of coordinates matches MAX_COORDS
        if len(coords) > MAX_COORDS:
            coords = coords[:MAX_COORDS]
        elif len(coords) < MAX_COORDS:
            padding = np.zeros((MAX_COORDS - len(coords), 2))
            coords = np.vstack((coords, padding))

        image = torch.from_numpy(image).unsqueeze(0).float()
        coords = torch.from_numpy(coords).float()

        return image, coords

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def up_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 32)  # Reduced number of filters
        self.encoder2 = conv_block(32, 64)
        
        self.bottleneck = conv_block(64, 128)
        
        self.upconv2 = up_conv_block(128, 64)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = up_conv_block(64, 32)
        self.decoder1 = conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.fc_coords = nn.Linear(128 * 64 * 64, MAX_COORDS * 2)  # Adjusted for new bottleneck size

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e2, 2))
        
        # Decoder
        d2 = self.upconv2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        # Output
        out = self.out_conv(d1)
        
        # Flatten and pass through a fully connected layer for coordinate prediction
        coords = self.fc_coords(b.view(b.size(0), -1))
        
        return out, coords.view(coords.size(0), -1, 2)

# Adjust the training loop and loss function accordingly
def loss_function(recon_x, x, coords, target_coords):
    BCE = nn.BCEWithLogitsLoss()(recon_x, x)
    MSE = nn.MSELoss()(coords, target_coords)
    return BCE + MSE

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
                recon_batch, coords = model(data)
                loss = loss_function(recon_batch, data, coords, target_coords)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                pbar.update(1)

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
        
    # Save the model after training
    model_save_path = 'unet_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Adjust main function accordingly
def main():
    data_dir = 'C:/Users/Jackson/Documents/mt_data/postprocessed'  # Replace with your data directory
    batch_size = 16  # Reduced batch size
    learning_rate = 2e-4
    epochs = 10
    num_workers = 4  # Number of worker processes for data loading

    dataset = MicrotubuleDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = UNet().cuda() if torch.cuda.is_available() else UNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, epochs=epochs)

if __name__ == "__main__":
    main()
