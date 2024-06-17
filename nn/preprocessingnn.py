import os
import cv2
import numpy as np
import argparse
from skimage.morphology import skeletonize
from tqdm import tqdm

def normalize_image(image):
    return image / 255.0

def resize_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def denoise_image(image):
    image = (image * 255).astype(np.uint8)  # Convert to uint8 before denoising
    if len(image.shape) == 2:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, h=3, templateWindowSize=21, searchWindowSize=7)
    else:
        raise ValueError("Unsupported image format for denoising")

def brighten_and_increase_contrast(image, brighten_factor=1.2, contrast_factor=1.2):
    image = image.astype(np.float32)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brighten_factor * 50)
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image

def preprocess_image(image, target_size=(512, 512)):
    image = normalize_image(image)
    image = resize_image(image, target_size)
    image = denoise_image(image)
    image = brighten_and_increase_contrast(image, brighten_factor=1.2, contrast_factor=3)
    return image

def save_preprocessed_image(image, filename):
    cv2.imwrite(filename, image)  # Image is already in uint8 format

def crop_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def preprocess_images_in_directory(input_directory, output_directory, bbox, original_size, target_size=(512, 512)):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    # Exclude the first frame
    filenames = filenames[1:]

    with tqdm(total=len(filenames), desc='Processing images') as pbar:
        for filename in filenames:
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            if os.path.exists(output_path):
                pbar.update(1)
                continue  # Skip already processed images

            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Resize image to match the selected bounding box
                resized_image = resize_image(image, target_size)
                cropped_image = crop_image(resized_image, bbox)
                preprocessed_image = preprocess_image(cropped_image, target_size)
                save_preprocessed_image(preprocessed_image, output_path)
            else:
                print(f"Error: Unable to load image {filename}")

            pbar.update(1)

def rotate_coordinates(x, y, angle, img_width, img_height):
    if angle == 90:
        return img_height - y, x
    elif angle == 180:
        return img_width - x, img_height - y
    elif angle == 270:
        return y, img_width - x
    else:
        return x, y

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

def rescale_coordinates_bbox(coords, bbox, target_size=(512, 512)):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    scale_x = target_size[0] / bbox_w
    scale_y = target_size[1] / bbox_h
    rescaled_coords = []
    for coord in coords:
        frame, coord_num, x, y, val = coord
        # Translate coordinates based on the bounding box
        x = x - bbox_x
        y = y - bbox_y
        # Scale coordinates
        new_x = x * scale_x
        new_y = y * scale_y
        if new_x > 512:
            print(new_x)
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
    parser.add_argument('--rotation_angle', type=int, default=0, help='Angle to rotate coordinates (0, 90, 180, 270)')

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    coords_file = args.coords_file
    rotation_angle = args.rotation_angle

    # Display the first and last frame and let the user select a bounding box
    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    if len(filenames) >= 2:
        # Skip the first frame
        first_image_path = os.path.join(input_directory, filenames[1])
        last_image_path = os.path.join(input_directory, filenames[-1])

        first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
        last_image = cv2.imread(last_image_path, cv2.IMREAD_GRAYSCALE)

        if first_image is not None and last_image is not None:
            first_image_resized = resize_image(first_image, (512, 512))
            last_image_resized = resize_image(last_image, (512, 512))
            
            combined_image = np.hstack((first_image_resized, last_image_resized))
            bbox = cv2.selectROI("Select ROI", combined_image, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            if bbox is not None:
                # Draw the bounding box on both images for confirmation
                x, y, w, h = bbox
                combined_image_with_bbox = combined_image.copy()
                if x >= 512:  # Selected region is in the second image
                    x1 = x - 512
                    cv2.rectangle(combined_image_with_bbox[:, :512], (x1, y), (x1 + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(combined_image_with_bbox[:, 512:], (x1, y), (x1 + w, y + h), (255, 0, 0), 2)
                else:
                    cv2.rectangle(combined_image_with_bbox[:, :512], (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.rectangle(combined_image_with_bbox[:, 512:], (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.imshow("Confirm ROI", combined_image_with_bbox)
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()

                if key == ord('y') or key == 13:  # If user confirms with 'y' or Enter
                    if x >= 512:  # Adjust x if the selection is from the second image
                        x = x - 512
                    preprocess_images_in_directory(input_directory, output_directory, (x, y, w, h), first_image.shape, target_size=(512, 512))

                    original_size = first_image.shape if first_image is not None else (512, 512)

                    coords = read_coordinates(coords_file)
                    # Filter out coordinates belonging to the first frame
                    coords = [coord for coord in coords if coord[0] != 1]
                    coords = rescale_coordinates(coords, original_size=original_size, target_size=(512, 512))
                    rescaled_coords = rescale_coordinates_bbox(coords, (x, y, w, h), target_size=(512, 512))
                    output_coords_file = os.path.join(output_directory, 'rescaled_coords.txt')
                    with open(output_coords_file, 'w') as file:
                        for coord in rescaled_coords:
                            file.write(f"{coord[0]}\t{coord[1]}\t{coord[2]:.6f}\t{coord[3]:.6f}\t{coord[4]}\n")
                    print(f"Saved rescaled coordinates to {output_coords_file}")
                else:
                    print("Bounding box selection canceled.")
            else:
                print("No bounding box selected.")
        else:
            print(f"Error: Unable to load the first or last image {first_image_path} or {last_image_path}")
    else:
        print(f"Not enough valid image files found in {input_directory}")

if __name__ == '__main__':
    main()
