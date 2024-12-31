import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def read_coordinates(file_path):
    frames = {}
    bbox = None
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 1:
                bbox = line[line.find("(")+1:line.find(")")]
                bbox = tuple(map(int, bbox.split(', ')))
            if len(parts) == 5:
                frame_id = int(parts[0])
                x = float(parts[2])
                y = float(parts[3])
                
                if frame_id not in frames:
                    frames[frame_id] = []
                frames[frame_id].append((x, y))
    return frames, bbox

def load_images_from_directory(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])
    print("loading images...")
    images = [cv2.imread(file) for file in tqdm(files)]
    return images

def show_image_and_coordinates(frame_num, coords, images, bbox):
    image = images[frame_num]
    image = image[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    for coord in coords[frame_num]:
        image[coord[0],coord[1]] = [0,0,255]
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    file_path = '/home/yuming/Documents/dev/python/projects/microtubule-tracking-2/rough_coordinates.txt'
    images_path = '/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm'
    coords, bbox = read_coordinates(file_path)
    images = load_images_from_directory(images_path)
    show_image_and_coordinates(frame_num = 1, coords= coords, images = images, bbox = bbox)
