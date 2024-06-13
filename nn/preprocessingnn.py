import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm


def normalize_image(image):
    return image / 255.0


def resize_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def denoise_image(image):
    image = (image * 255).astype(np.uint8)  # Convert to uint8 before denoising
    if len(image.shape) == 2:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        raise ValueError("Unsupported image format for denoising")


def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def remove_small_objects(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background
    cleaned_image = np.zeros_like(image)
    cleaned_image[labels == largest_label] = image[labels == largest_label]

    return cleaned_image


def preprocess_image(image, target_size=(512, 512)):
    image = normalize_image(image)
    image = resize_image(image, target_size)
    image = denoise_image(image)
    if len(image.shape) == 2:  # Grayscale image
        image = enhance_contrast(image)
    image = remove_small_objects(image)
    return image


def save_preprocessed_image(image, filename):
    cv2.imwrite(filename, image)  # Image is already in uint8 format


def preprocess_images_in_directory(input_directory, output_directory, target_size=(512, 512)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    with tqdm(total=len(filenames), desc='Processing images') as pbar:
        for filename in filenames:
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            if os.path.exists(output_path):
                pbar.update(1)
                continue  # Skip already processed images

            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                preprocessed_image = preprocess_image(image, target_size)
                save_preprocessed_image(preprocessed_image, output_path)
            else:
                print(f"Error: Unable to load image {filename}")

            pbar.update(1)


def rescale_coordinates(coords, original_size, target_size=(512, 512)):
    scale_x = target_size[0] / original_size[1]
    scale_y = target_size[1] / original_size[0]
    rescaled_coords = []
    for coord in coords:
        frame, coord_num, x, y, val = coord
        new_x = x * scale_x
        new_y = y * scale_y
        rescaled_coords.append((frame, coord_num, new_x, new_y, val))
    return rescaled_coords


def read_coordinates(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    coords = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.split()
        if len(parts) == 5:
            frame, coord_num, x, y, val = map(float, parts)
            coords.append((int(frame), int(coord_num), x, y, val))
    return coords


def main():
    parser = argparse.ArgumentParser(description="Preprocess microtubule images and coordinates.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed images.')
    parser.add_argument('--coords_file', type=str, required=True, help='File containing coordinates for training data.')

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    coords_file = args.coords_file

    preprocess_images_in_directory(input_directory, output_directory, target_size=(512, 512))

    image_example_path = next((os.path.join(input_directory, f) for f in os.listdir(input_directory)
                               if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')), None)

    if image_example_path:
        original_image = cv2.imread(image_example_path, cv2.IMREAD_GRAYSCALE)
        original_size = original_image.shape if original_image is not None else (512, 512)

        coords = read_coordinates(coords_file)
        rescaled_coords = rescale_coordinates(coords, original_size, target_size=(512, 512))

        output_coords_file = os.path.join(output_directory, 'rescaled_coords.txt')
        with open(output_coords_file, 'w') as file:
            for coord in rescaled_coords:
                file.write(f"{coord[0]}\t{coord[1]}\t{coord[2]:.6f}\t{coord[3]:.6f}\t{coord[4]}\n")
        print(f"Saved rescaled coordinates to {output_coords_file}")


if __name__ == '__main__':
    main()



