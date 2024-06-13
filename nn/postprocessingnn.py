import os
import cv2
import numpy as np
import torch
import argparse
from torchvision import transforms

def process_image(image, device, rotation_step=10):
    transformed_images = []

    # Convert image to PyTorch tensor
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Apply rotations at specified intervals
    for angle in range(0, 360, rotation_step):
        rotation_matrix = cv2.getRotationMatrix2D((256, 256), angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        transformed_images.append(rotated_image)
    
    # Apply other transformations
    for image in transformed_images.copy():
        # Scaling (Zoom in/out)
        for scale in [0.8, 1.2]:
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_image = cv2.copyMakeBorder(scaled_image, 
                                              top=(512-scaled_image.shape[0])//2, 
                                              bottom=(512-scaled_image.shape[0])//2, 
                                              left=(512-scaled_image.shape[1])//2, 
                                              right=(512-scaled_image.shape[1])//2, 
                                              borderType=cv2.BORDER_CONSTANT, value=0)
            transformed_images.append(scaled_image[:512, :512])

        # Translation (Shift)
        for shift in [20, -20]:
            M = np.float32([[1, 0, shift], [0, 1, shift]])
            translated_image = cv2.warpAffine(image, M, (512, 512), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            transformed_images.append(translated_image)

        # Horizontal and Vertical Flips
        flipped_image_h = cv2.flip(image, 1)
        transformed_images.append(flipped_image_h)

        flipped_image_v = cv2.flip(image, 0)
        transformed_images.append(flipped_image_v)

    return transformed_images

def process_images_in_directory(input_dir, output_dir, device, rotation_step):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Process the image
            processed_images = process_image(image, device, rotation_step)
            
            # Save processed images
            base_filename, ext = os.path.splitext(filename)
            for i, processed_image in enumerate(processed_images):
                output_path = os.path.join(output_dir, f"{base_filename}_transformed_{i}{ext}")
                cv2.imwrite(output_path, processed_image)

def main():
    parser = argparse.ArgumentParser(description="Process grayscale, time-series, 512x512 images of microtubules.")
    parser.add_argument("input_dir", type=str, help="Directory of input images")
    parser.add_argument("output_dir", type=str, help="Directory to save processed images")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing if available")
    parser.add_argument("--rotation_step", type=int, default=10, help="Step size for rotations in degrees (default is 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    process_images_in_directory(args.input_dir, args.output_dir, device, args.rotation_step)

if __name__ == "__main__":
    main()


