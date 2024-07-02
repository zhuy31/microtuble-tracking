import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

bbox_start = (0, 0)
dragging = False
bbox = None
combined_image = None
img_width = 0

def crop_image(image, bbox, display_size = 1024):
    x, y, w, h = bbox
    x, w = x*image.shape[0]/display_size, w*image.shape[0]/display_size
    y, h = y*image.shape[0]/display_size, h*image.shape[0]/display_size
    x, y, w, h = int(x), int(y), int(w), int(h)
    return image[y:y+h, x:x+w]

def save_preprocessed_image(image, filename):
    if(len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, image)

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def connected_components(image, threshold=20):
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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    image = image / 255.0
    
    image = resize_image(image, target_size)

    image = (image * 255).astype(np.uint8) 
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    image = connected_components(image)

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

def preprocess_images_in_directory(input_directory, output_directory, bbox, target_size=(256, 256), test_dir=None, display_size = 1024):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if test_dir and not os.path.exists(test_dir):
        os.makedirs(test_dir)

    filenames = sorted([f for f in os.listdir(input_directory)
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')])

    filenames = filenames[1:]

    with tqdm.tqdm(total=len(filenames), desc='Processing images') as pbar:
        i = 0
        for filename in filenames:
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            if os.path.exists(output_path):
                pbar.update(1)
                continue
            image = cv2.imread(input_path)
            if image is not None:
                cropped_image = crop_image(image, bbox, display_size = display_size)
                image = preprocess_image(cropped_image, target_size)
                if test_dir is not None and cropped_image is not None:
                    test_output_path = os.path.join(test_dir, f'{i+1}.jpg')
                    cv2.imwrite(test_output_path, cropped_image)

                save_preprocessed_image(resize_image(image, target_size), output_path)
                
            else:
                print(f"Error: Unable to load image {filename}")
            pbar.update(1)

def main():
    display_size = sample_images_from_folder('C:/Users/Jackson/Downloads/MT_plus_Tracking/MT_plus_Tracking/5_20/MT10_2/MT10_30min_200x_1500_138_146pm')
    preprocess_images_in_directory('C:/Users/Jackson/Downloads/MT_plus_Tracking/MT_plus_Tracking/5_20/MT10_2/MT10_30min_200x_1500_138_146pm','C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2',bbox, display_size=display_size)

if __name__ == '__main__':
    main()
