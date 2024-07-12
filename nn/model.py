import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


# Define the maximum number of coordinates
MAX_COORDS = 400

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

        self.encoder1 = conv_block(in_channels, 32)
        self.encoder2 = conv_block(32, 64)
        
        self.bottleneck = conv_block(64, 128)
        
        self.upconv2 = up_conv_block(128, 64)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = up_conv_block(64, 32)
        self.decoder1 = conv_block(64, 32)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.fc_coords = nn.Linear(128 * 64 * 64, MAX_COORDS * 2)

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

def load_model(model_path):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_coordinates(model, image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(image, (512, 512))  # Resize for visualization
    image = cv2.resize(image, (256, 256))  # Resize for model input
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    with torch.no_grad():
        _, coords = model(image)
    
    coords = coords.squeeze().numpy()
    return coords, original_image

def overlay_coordinates(image, coords, color=(0, 0, 255)):
    for coord in coords:
        x, y = int(coord[0]), int(coord[1])
        if x >= 0 and y >= 0 and x < image.shape[1] and y < image.shape[0]:
            cv2.circle(image, (x, y), 2, color, -1)
    return image

def main(image_path):
    model_path = 'unet_model.pth'
    model = load_model(model_path)
    coords, original_image = predict_coordinates(model, image_path)
    overlay_image = overlay_coordinates(original_image, coords)
    
    cv2.imshow('Overlay Image', overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Optionally save the overlay image
    output_path = 'overlay_image.png'
    cv2.imwrite(output_path, overlay_image)
    print(f"Overlay image saved to {output_path}")

if __name__ == "__main__":
    import sys
    main('C:/Users/Jackson/Documents/mt_data/preprocessed/imageset1/MT7_30min_100x_443_453pm_1500_t0166.jpg')
