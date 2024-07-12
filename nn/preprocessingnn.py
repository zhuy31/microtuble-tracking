import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize
bbox_start = (0, 0)
dragging = False
bbox = None
combined_image = None
img_width = 0

def crop_image(original_image, bbox, display_size=1024):
    scale_x = original_image.shape[1] / display_size
    scale_y = original_image.shape[0] / display_size
    x, y, w, h = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
    return original_image[y:y+h, x:x+w]

def save_preprocessed_image(image, filename, target_size = None):
    if(len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if target_size is not None:
        image = resize_image(image, target_size=target_size)
    cv2.imwrite(filename, image)

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def connected_components(image, threshold=0):
    image = cv2.fastNlMeansDenoising(image, None, 20, 7, 21)
    
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned_image = np.zeros_like(image)

    max_area = -1
    max_pointer = None
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max_area:
            max_pointer = i
            max_area = stats[i, cv2.CC_STAT_AREA]

    cleaned_image[labels == max_pointer] = image[labels == max_pointer]
    return cleaned_image

def preprocess_image(image,  target_size = (256,256)):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.fastNlMeansDenoisingColored(image, None, 30, 10, 7, 21)
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    image = connected_components(image)

    if target_size is not None:
        image = resize_image(image, target_size)
    return image

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

            # Draw the bounding box on all images
            for i in range(len(param)):
                x1 = bbox[0] - i * img_width if bbox[0] >= i * img_width else bbox[0]
                if 0 <= x1 < img_width:
                    cv2.rectangle(combined_image_with_bbox[:, i*img_width:(i+1)*img_width], 
                                  (x1, bbox[1]), (x1 + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
            cv2.imshow("Select ROI", combined_image_with_bbox)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False




def display_and_select_bounding_boxes(images, target_size=(256, 256)):
    global bbox_start, dragging, bbox, combined_image, img_width

    if len(images) < 1:
        print("No images provided.")
        return None

    resized_images = [resize_image(img, target_size) for img in images]

    combined_image = np.hstack(resized_images)
    
    img_width = combined_image.shape[1] // len(images)
    
    window_name = "Select ROI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, combined_image.shape[1] * 2, combined_image.shape[0] * 2)
    cv2.setMouseCallback(window_name, mouse_callback, param=images)

    cv2.imshow(window_name, combined_image)
    while True:
        key = cv2.waitKey(1)
        if key == ord('y') or key == 13:
            cv2.destroyAllWindows()
            cropped_images = [img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] for img in resized_images]
            return cropped_images, bbox
        elif key == ord('n') or key == 27:
            cv2.destroyAllWindows()
            print("Bounding box selection canceled.")
            return None

def plot_images_side_by_side(images, titles=None):
    num_images = len(images)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 10))
    
    if titles is None:
        titles = [''] * num_images

    for i, (image, title) in enumerate(zip(images, titles)):
        # Original image on the top row
        axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        # Preprocessed image on the bottom row
        preprocessed_image = preprocess_image(image)
        axes[1, i].imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title(f"Preprocessed {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def sample_images_from_folder(directory, target_size=(1024, 1024)):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])

    if len(image_files) < 5:
        print("Not enough images in the directory.")
        return

    interval = len(image_files) // 4 -1
    selected_images = [image_files[i * interval] for i in range(5)]

    images = [cv2.imread(os.path.join(directory, img)) for img in selected_images]

    cropped_images, selected_bbox = display_and_select_bounding_boxes(images, target_size=target_size)
    if cropped_images:
        plot_images_side_by_side(cropped_images, titles=selected_images)
        cv2.waitKey(0)
    return target_size[0]

def read_coordinates(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    coords = []
    invalid_lines = 0
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.split()
        if len(parts) == 5:
            frame, coord_num, x, y, val = map(float, parts)
            coords.append((int(frame), int(coord_num), x, y, val))
        else:
            invalid_lines = invalid_lines+1
    return coords

def crop_and_rescale_coordinates(image, bbox, coords, display_size=1024, target_size=(256, 256)):
    scale_x = image.shape[1] / display_size
    scale_y = image.shape[0] / display_size
    x, y, w, h = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
    cropped_image = image[y:y+h, x:x+w]
    rescale_x = target_size[0] / w
    rescale_y = target_size[1] / h
    rescaled_coords = [(frame, coord_num, (x_coord - x) * rescale_x, (y_coord - y) * rescale_y, val)
                       for frame, coord_num, x_coord, y_coord, val in coords]
    return cropped_image, rescaled_coords

def process_single_image(input_path, output_path, output_directory, coords, bbox, target_size, display_size, test_dir=None, manual_tracking=False):
    image = cv2.imread(input_path)
    
    if image is not None and manual_tracking is False:
        cropped_image, rescaled_coords = crop_and_rescale_coordinates(image, bbox, coords, display_size=display_size, target_size=target_size)
        preprocessed_image = preprocess_image(cropped_image, target_size)
        if test_dir is not None:
            test_output_path = os.path.join(test_dir, os.path.basename(output_path))
            cv2.imwrite(test_output_path, crop_image(image, bbox))
        save_preprocessed_image(preprocessed_image, output_path,target_size=target_size)
        output_coords_file = os.path.join(output_directory, 'rescaled_coords.e')
        with open(output_coords_file, 'w') as file:
            for coord in rescaled_coords:
                if coord[0] != 1:
                    file.write(f"{coord[0]-1}\t{coord[1]}\t{coord[2]:.6f}\t{coord[3]:.6f}\t{coord[4]}\n")
    elif image is not None:
        cropped_image, __ = crop_and_rescale_coordinates(image, bbox, coords, display_size=display_size, target_size=target_size)
        if test_dir is not None:
            test_output_path = os.path.join(test_dir, os.path.basename(output_path))
            cv2.imwrite(test_output_path, crop_image(image, bbox))
        save_preprocessed_image(preprocess_image(cropped_image, target_size = None), output_path,target_size=None)
        
    else:
        print(f"Error: Unable to load image {input_path}")

def preprocess_images_in_directory(input_directory, output_directory, coords_file, bbox, target_size=(256, 256), test_dir=None, display_size=1024, use_multithreading=False, max_workers=4, manual_tracking = False):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if test_dir and not os.path.exists(test_dir):
        os.makedirs(test_dir)
    print(f"test_dir = {test_dir}")
    filenames = sorted([f for f in os.listdir(input_directory) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])
    filenames = filenames[1:]
    coords = read_coordinates(coords_file)

    with tqdm.tqdm(total=len(filenames), desc='Processing images') as pbar:
        if use_multithreading:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for filename in filenames:
                    input_path = os.path.join(input_directory, filename)
                    output_path = os.path.join(output_directory, filename)
                    if os.path.exists(output_path):
                        pbar.update(1)
                        continue
                    futures.append(executor.submit(process_single_image, input_path, output_path, output_directory, coords, bbox, target_size, display_size, test_dir=test_dir, manual_tracking=manual_tracking))
                for future in futures:
                    future.result()  # Ensure all futures are completed
                    pbar.update(1)
        else:
            for filename in filenames:
                input_path = os.path.join(input_directory, filename)
                output_path = os.path.join(output_directory, filename)
                if os.path.exists(output_path):
                    pbar.update(1)
                    continue
                process_single_image(input_path, output_path, coords, bbox, target_size, display_size, test_dir)
                pbar.update(1)

    

def main(use_multithreading=False, max_workers=4):
    display_size = sample_images_from_folder('C:/Users/Jackson/Downloads/MT_plus_Tracking/MT_plus_Tracking/5_20/MT10_2/MT10_30min_200x_1500_138_146pm')
    preprocess_images_in_directory(
        'C:/Users/Jackson/Downloads/MT_plus_Tracking/MT_plus_Tracking/5_20/MT10_2/MT10_30min_200x_1500_138_146pm',
        'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2',
        'c:/Users/Jackson/Downloads/MT_plus_Tracking/MT_plus_Tracking/5_20/MT10_2/MT10_2_snake/MT10_1113.txt',
        bbox,
        display_size=display_size,
        use_multithreading=use_multithreading,
        max_workers=max_workers,
        manual_tracking= True,
        test_dir= 'C:/Users/Jackson/Documents/mt_data/experimental'
    )

if __name__ == '__main__':
    main(use_multithreading=True, max_workers=4)
