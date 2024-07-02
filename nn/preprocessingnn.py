import os
import cv2
import numpy as np
import argparse
from skimage.morphology import skeletonize
from tqdm import tqdm

def normalize_image(image):
    return image / 255.0

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def denoise_image(image):
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, h=3, templateWindowSize=21, searchWindowSize=7)
    else:
        raise ValueError("Unsupported image format for denoising")

def brighten_and_increase_contrast(image, brighten_factor=1.2, contrast_factor=1.2):
    image = image.astype(np.float32)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brighten_factor * 50)
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    clahe_channels = [clahe.apply(channel) for channel in channels]
    return cv2.merge(clahe_channels)

def calculate_contour_std_dev(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean, std_dev = cv2.meanStdDev(image, mask=mask)
    return std_dev[0][0]

def calculate_contour_light_density(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(image, mask=mask)[0]  # Get the mean light density
    return mean_val

def remove_all_but_biggest_component(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
    largest_component_mask = (labels == largest_component_label).astype(np.uint8) * 255
    largest_component_mask_colored = cv2.merge([largest_component_mask] * 3)
    largest_component_image = cv2.bitwise_and(image, largest_component_mask_colored)
    
    return largest_component_image

def choosemethod(image, previous_contours=None, target_size=(256, 256), iteration=0):
    pass


def save_preprocessed_image(image, filename):
    if(len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, image)

def crop_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def calculate_average_difference(image, image_list):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    differences = [np.mean(cv2.absdiff(image, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) for img in image_list]
    return np.mean(differences)

def draw_contour_on_most_different_color(image, previous_images):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    
    best_score = -1
    best_contour = None
    optimal_penalty = -1

    for i, channel in enumerate([h, s, v]):
        _, binary_image = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image.shape[0] * image.shape[1] / 16:
                area = 0
            contour_std_dev = calculate_contour_std_dev(channel, contour)
            light_density = calculate_contour_light_density(channel, contour)
            
            penalty = 0
            if previous_images:
                masked_image = cv2.bitwise_and(image, image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
                penalty = calculate_average_difference(masked_image, previous_images)

            if contour_std_dev > 0:
                score = (contour_std_dev * area) + light_density - penalty
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_contour = contour
                optimal_penalty = penalty
    
    if best_contour is not None:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [best_contour], -1, 255, -1)
        kernel_size = max(1, int(image.shape[0] / 256))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.erode(mask, kernel, iterations=1)
        masked_image = cv2.bitwise_and(image, image, mask=dilated_mask)
        return masked_image, best_contour, optimal_penalty
    
    return image, optimal_penalty


def preprocess_image_method_1(image,  previous_images, target_size=(256, 256)):
    image = normalize_image(image)
    image = resize_image(image, target_size)
    image = denoise_image(image)
    image = brighten_and_increase_contrast(image)
    image = apply_clahe(image)
    image, best_contour, optimal_penalty = draw_contour_on_most_different_color(image, previous_images)
    image = remove_all_but_biggest_component(image)
    return image, optimal_penalty

def preprocess_image_method_2(image, previous_images, target_size = (256,256), min_size = 100):
    # Normalize
    penalty = -2
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    image = image / 255.0
    
    image = resize_image(image, target_size)

    image = (image * 255).astype(np.uint8) 
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned_image = np.zeros_like(image)
    
    # Loop through all connected components
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = image[labels == i]
    
    if previous_images:
        penalty = calculate_average_difference(cv2.threshold(cleaned_image, 5, cv2.THRESH_BINARY), previous_images)

    return cleaned_image, penalty

def preprocess_image(image, previous_images, target_size=(256, 256), iteration = 0):
    image1, pen1 = preprocess_image_method_1(image, previous_images)
    image2, pen2 = preprocess_image_method_2(image, previous_images)

    if pen2 <= pen1 or iteration < 200:
        return image2
    else:
        return image1

def preprocess_images_in_directory(input_directory, output_directory, bbox, original_size, target_size=(256, 256), test_dir=None):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if test_dir and not os.path.exists(test_dir):
        os.makedirs(test_dir)

    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    filenames = filenames[1:]
    
    previous_contours = []
    previous_images = []

    with tqdm(total=len(filenames), desc='Processing images') as pbar:
        i = 0
        for filename in filenames:
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            if os.path.exists(output_path):
                pbar.update(1)
                continue

            image = cv2.imread(input_path)
            if image is not None:
                resized_image = resize_image(image, target_size)
                cropped_image = crop_image(resized_image, bbox)

                if test_dir is not None and cropped_image is not None:
                    test_output_path = os.path.join(test_dir, f'{i+1}.jpg')
                    cv2.imwrite(test_output_path, cropped_image)

                i += 1
                preprocessed_image = preprocess_image(cropped_image, previous_images, target_size, i)
                save_preprocessed_image(preprocessed_image, output_path)
                
                if preprocessed_image is not None:
                    previous_contours.append(preprocessed_image)
                    if len(previous_images) > 8:
                        previous_images.pop(0)
            else:
                print(f"Error: Unable to load image {filename}")

            pbar.update(1)



def rescale_coordinates(coords, original_size, target_size=(256, 256)):
    scale_x = target_size[0] / original_size[1]
    scale_y = target_size[1] / original_size[0]
    rescaled_coords = []
    for coord in coords:
        frame, coord_num, x, y, val = coord
        new_x = x * scale_x
        new_y = y * scale_y
        rescaled_coords.append((frame, coord_num, new_x, new_y, val))
    
    return rescaled_coords

def rescale_coordinates_bbox(coords, bbox, target_size=(256, 256)):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    scale_x = target_size[0] / bbox_w
    scale_y = target_size[1] / bbox_h
    rescaled_coords = []
    for coord in coords:
        frame, coord_num, x, y, val = coord
        x = x - bbox_x
        y = y - bbox_y
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

# Add this new function to handle mouse events and update bounding boxes in real-time
def mouse_callback(event, x, y, flags, param):
    global bbox_start, dragging, bbox, combined_image, img_width

    if event == cv2.EVENT_LBUTTONDOWN:
        bbox_start = (x, y)
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            combined_image_with_bbox = combined_image.copy()
            bbox_end = (x, y)
            bbox = (bbox_start[0], bbox_start[1], bbox_end[0] - bbox_start[0], bbox_end[1] - bbox_start[1])

            # Draw the bounding box on all three images
            for i in range(3):
                x1 = bbox[0] - i * img_width if bbox[0] >= i * img_width else bbox[0]
                if 0 <= x1 < img_width:
                    cv2.rectangle(combined_image_with_bbox[:, i*img_width:(i+1)*img_width], (x1, bbox[1]), (x1 + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            cv2.imshow("Select ROI", combined_image_with_bbox)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# Replace the existing bounding box selection function with this one
def display_and_select_bounding_boxes(input_directory, original_size, target_size=(256, 256)):
    global bbox_start, dragging, bbox, combined_image, img_width

    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    if len(filenames) >= 3:
        first_image_path = os.path.join(input_directory, filenames[1])
        middle_image_path = os.path.join(input_directory, filenames[len(filenames) // 2])
        last_image_path = os.path.join(input_directory, filenames[-1])
        
        first_image = cv2.imread(first_image_path)
        middle_image = cv2.imread(middle_image_path)
        last_image = cv2.imread(last_image_path)
        
        if first_image is not None and middle_image is not None and last_image is not None:
            first_image_resized = resize_image(first_image, target_size)
            middle_image_resized = resize_image(middle_image, target_size)
            last_image_resized = resize_image(last_image, target_size)
            
            combined_image = np.hstack((first_image_resized, middle_image_resized, last_image_resized))
            img_width = combined_image.shape[1] // 3
            
            window_name = "Select ROI"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, combined_image.shape[1] * 2, combined_image.shape[0] * 2)
            cv2.setMouseCallback(window_name, mouse_callback)

            cv2.imshow(window_name, combined_image)
            while True:
                key = cv2.waitKey(1)
                if key == ord('y') or key == 13:
                    cv2.destroyAllWindows()
                    return bbox
                elif key == ord('n') or key == 27:
                    cv2.destroyAllWindows()
                    print("Bounding box selection canceled.")
                    return None
        else:
            print(f"Error: Unable to load one or more images: {first_image_path}, {middle_image_path}, {last_image_path}")
            return None
    else:
        print(f"Not enough valid image files found in {input_directory}")
        return None

# Initialize global variables
bbox_start = (0, 0)
dragging = False
bbox = None
combined_image = None
img_width = 0

def main():
    parser = argparse.ArgumentParser(description="Preprocess microtubule images and coordinates.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed images.')
    parser.add_argument('--coords_file', type=str, required=True, help='File containing coordinates for training data.')
    parser.add_argument('--test_dir', type=str, default=None, help='dir to output cropped images ONLY')

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir
    coords_file = args.coords_file


    original_size = (512, 512)  # Example size, adjust accordingly

    bbox = display_and_select_bounding_boxes(input_directory, original_size, target_size=(256, 256))
    if bbox:
        preprocess_images_in_directory(input_directory, output_directory, bbox, original_size, target_size=(256, 256), test_dir=args.test_dir)
        coords = read_coordinates(coords_file)
        coords = [coord for coord in coords if coord[0] != 1]
        coords = rescale_coordinates(coords, original_size=original_size, target_size=(256, 256))
        rescaled_coords = rescale_coordinates_bbox(coords, bbox, target_size=(256, 256))
        output_coords_file = os.path.join(output_directory, 'rescaled_coords.txt')
        with open(output_coords_file, 'w') as file:
            for coord in rescaled_coords:
                file.write(f"{coord[0]}\t{coord[1]}\t{coord[2]:.6f}\t{coord[3]:.6f}\t{coord[4]}\n")
        print(f"Saved rescaled coordinates to {output_coords_file}")
    else:
        print("Bounding box selection canceled.")

if __name__ == '__main__':
    main()
