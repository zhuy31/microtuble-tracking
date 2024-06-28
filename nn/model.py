import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from vae_model import VAE

# Define the maximum number of coordinates
MAX_COORDS = 400

def load_model(model_path, device):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_coordinates(model, image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512, 512))
    image = transform(image).unsqueeze(0).to(device).float() / 255.0  # Normalize and add batch dimension

    with torch.no_grad():
        _, coords, _, _ = model(image)
    
    return image.cpu().squeeze().numpy(), coords.cpu().squeeze().numpy()

def plot_image_with_coords(image, coords, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    # Plot predicted coordinates
    plt.scatter(coords[:, 0], coords[:, 1], c='red', label='Predicted Coordinates', s=5)
    
    plt.legend()
    plt.title('Predicted Coordinates Overlay')
    
    plt.savefig(output_path)
    plt.show()

def main(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'vae_model.pth'  # Path to the saved model
    model = load_model(model_path, device)
    
    image, coords = predict_coordinates(model, image_path, device)
    
    output_path = 'output_with_coords.png'
    plot_image_with_coords(image, coords, output_path)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main('C:/Users/Jackson/Documents/mt_data/preprocessed/imageset1/MT7_30min_100x_443_453pm_1500_t0002.jpg')
