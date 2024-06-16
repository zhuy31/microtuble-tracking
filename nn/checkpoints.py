import os
import cv2
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt

def load_coordinates(txt_file, image_number):
    coords = []
    with open(txt_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if int(parts[0]) == image_number:
                coords.append([float(parts[2]), float(parts[3])])
    return np.array(coords)

def plot_points_on_image(image, coords):
    plt.imshow(image, cmap='gray')
    plt.scatter(coords[:, 0], coords[:, 1], s=10, c='red', marker='o')
    plt.axis('off')
    plt.show()

def process_images(input_dir):
    image_files = glob(os.path.join(input_dir, "*.jpg"))
    txt_files = glob(os.path.join(input_dir, "*.txt"))
    
    if len(image_files) >= 2 and len(txt_files) > 0:
        second_image_file = image_files[1]
        first_txt_file = txt_files[0]
        
        image = cv2.imread(second_image_file, cv2.IMREAD_GRAYSCALE)
        coords = load_coordinates(first_txt_file, 2)  # 2 for the second image
        plot_points_on_image(image, coords)
    else:
        print("No image files or .txt files found in the directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot points on the second image in the dataset.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images and points files')

    args = parser.parse_args()

    process_images(args.input_dir)
