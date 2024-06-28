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
        subfolders = subfolders
        print("subfolders: {subfolders}")
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
                
                self.image_paths.extend(images)
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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 32 * 32, 512)
        self.fc_logvar = nn.Linear(256 * 32 * 32, 512)
        self.fc_decode = nn.Linear(512, 256 * 32 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.fc_coords = nn.Linear(512, MAX_COORDS * 2)  # Produces MAX_COORDS coordinate pairs

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 32, 32)
        return self.decoder(h)

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


def plot_and_save_image(image, true_coords, pred_coords, epoch, filename, output_dir='output_images'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image.squeeze(), cmap='gray')
    
    true_coords = true_coords.cpu().numpy()
    pred_coords = pred_coords.cpu().detach().numpy()
    
    # Plot true coordinates
    plt.scatter(true_coords[:, 0], true_coords[:, 1], c='blue', label='True Coordinates', s=5)
    # Plot predicted coordinates
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', label='Predicted Coordinates', s=5)
    
    plt.legend()
    plt.title(f'Epoch {epoch + 1}')
    
    output_path = os.path.join(output_dir, f'epoch_{epoch + 1}_{filename}.png')
    plt.savefig(output_path)
    plt.close()

def train(model, dataloader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        last_data = None
        last_target_coords = None
        last_coords = None
        last_filename = None
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
                
                # Save last batch data for visualization
                last_data = data[-1].cpu()  # Save the last image in the batch
                last_target_coords = target_coords[-1].cpu()  # Save the corresponding target coordinates
                last_coords = coords[-1].cpu()  # Save the corresponding predicted coordinates
                last_filename = os.path.basename(dataloader.dataset.image_paths[batch_idx * data.size(0) + (data.size(0) - 1)])

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
        
        # Plot and save the image with coordinates overlayed
        plot_and_save_image(last_data, last_target_coords, last_coords, epoch, last_filename)


def main():
    data_dir = 'C:/Users/Jackson/Documents/mt_data/preprocessed'  # Replace with your data directory
    batch_size = 32
    learning_rate = 1e-3
    epochs = 10
    num_workers = 4  # Number of worker processes for data loading

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MicrotubuleDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = VAE().cuda() if torch.cuda.is_available() else VAE()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, dataloader, optimizer, epochs=epochs)

if __name__ == "__main__":
    main()
