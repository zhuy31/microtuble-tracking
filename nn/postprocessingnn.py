import os
import shutil
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

def fill_black_with_noise(image):
    mask = image == 0
    noise = np.random.normal(127, 127, image.shape).astype(np.uint8)
    image[mask] = noise[mask]
    return image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_with_noise = fill_black_with_noise(rotated)
    resized = cv2.resize(rotated_with_noise, (512, 512), interpolation=cv2.INTER_AREA)
    return resized, M

def reflect_image(image):
    return cv2.flip(image, 1)

def reflect_coordinates(coords, width):
    reflected_coords = coords.copy()
    reflected_coords[:, 2] = width - coords[:, 2]
    return reflected_coords

def transform_coordinates(coords, M):
    transformed_coords = coords.copy()
    for i in range(len(coords)):
        x, y = coords[i, 2], coords[i, 3]
        transformed_coords[i, 2] = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        transformed_coords[i, 3] = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return transformed_coords

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_sets = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    
    for image_set in image_sets:
        image_files = glob(os.path.join(image_set, "*.jpg"))
        txt_files = glob(os.path.join(image_set, "*.txt"))
        
        total_files = len(image_files) * 36 * 2  # 36 rotations * 2 (original and reflected)
        progress = tqdm(total=total_files, desc=f'Processing {os.path.basename(image_set)}')
        
        for angle in range(0, 360, 20):
            angle_dir = os.path.join(output_dir, os.path.basename(image_set) + f'_{angle}deg')
            angle_reflected_dir = os.path.join(output_dir, os.path.basename(image_set) + f'_{angle}deg_reflected')
            os.makedirs(angle_dir, exist_ok=True)
            os.makedirs(angle_reflected_dir, exist_ok=True)
            
            for image_file in image_files:
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                rotated_image, M = rotate_image(image, angle)
                reflected_image = reflect_image(rotated_image)
                
                cv2.imwrite(os.path.join(angle_dir, os.path.basename(image_file)), rotated_image)
                cv2.imwrite(os.path.join(angle_reflected_dir, os.path.basename(image_file)), reflected_image)
                
                progress.update(2)
            
            for txt_file in txt_files:
                coords = np.loadtxt(txt_file)
                transformed_coords = transform_coordinates(coords, M)
                np.savetxt(os.path.join(angle_dir, os.path.basename(txt_file)), transformed_coords, fmt='%d\t%d\t%.6f\t%.6f\t%.1f')
                
                reflected_coords = reflect_coordinates(transformed_coords, 512)  # assuming the image width is 512
                np.savetxt(os.path.join(angle_reflected_dir, os.path.basename(txt_file)), reflected_coords, fmt='%d\t%d\t%.6f\t%.6f\t%.1f')
        
        progress.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing image sets')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed images')

    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)
