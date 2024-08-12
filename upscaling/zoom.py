import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import fast_tsp

def connected_components2(image, threshold=0):
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

def process2(image, corner1, corner2, target_size = None):
    x1, y1 = corner1
    x2, y2 = corner2

    

    cropped_image = image[y1:y2, x1:x2]

    cropped_image = cv2.normalize(cropped_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image, None, 30, 10, 7, 21)
    if len(cropped_image.shape)==3:
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped_image = clahe.apply(cropped_image)
    cropped_image = cv2.normalize(cropped_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    cropped_image = connected_components2(cropped_image)
    
    image = np.zeros_like(image)
    image[y1:y2, x1:x2] = cropped_image

    return image


def resize(point1, point2, image, f):
    point1 = np.array(list(point1))
    point2 = np.array(list(point2))
    center = (point1+point2)*(1/2)
    point1 = f * (point1-center) + center
    point2 = f * (point2-center) + center
    point1 = [int(x) for x in point1]
    point2 = [int(x) for x in point2]

    point1 = [max(0,point1[0]),max(0,point1[1])]
    point2 = [min(image.shape[1]-1,point2[0]),min(image.shape[0]-1,point2[1])]

    return tuple(point1), tuple(point2)

def too_much_change(point1, point2, point3, point4, count):
    point1 = np.array(list(point1))
    point2 = np.array(list(point2))
    point3 = np.array(list(point3))
    point4 = np.array(list(point4))

    dist = np.linalg.norm(point3-point4)

    if (np.linalg.norm(point1-point3) > 0.15*dist or np.linalg.norm(point2-point4) > 0.15*dist) and count < 4:
        print("changed.")
        return point3, point4, count+1
    else:
        return point1, point2, 0


def draw_non_black_rectangle(image, prev_upper_left, prev_lower_right, count, original_image, factor = 1.5):

    non_black_pixels = np.where(image > 0)
    
    if non_black_pixels[0].size > 0:

        upper_left = (np.min(non_black_pixels[1]), np.min(non_black_pixels[0]))
        lower_right = (np.max(non_black_pixels[1]), np.max(non_black_pixels[0]))
        upper_left, lower_right = resize(upper_left, lower_right, image, factor)
        
    if prev_upper_left is not None and prev_lower_right is not None:
        upper_left, lower_right, count = too_much_change(upper_left, lower_right, prev_upper_left, prev_lower_right, count)

    image = process2(original_image, upper_left, lower_right)

    return image, upper_left, lower_right, count

def get_largest_connected_component(image):


    num_labels, labels = cv2.connectedComponents(image)
    
    if num_labels <= 1:
        return np.zeros_like(image)
    
    largest_label = 1 + np.argmax(np.bincount(labels.flat)[1:])
    largest_component = np.zeros_like(image)
    largest_component[labels == largest_label] = 255
    
    return largest_component

def process(image, upper_left, lower_right, count):
    image = cv2.normalize(
    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    original_image = image

    image = cv2.fastNlMeansDenoisingColored(image, None,15,10,7,21)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    _, image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
    image = cv2.erode(image, np.ones((4,4)), iterations=1) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(32,32))
    image = cv2.dilate(image, kernel, iterations=1) 

    image = get_largest_connected_component(image)

    image, upper_left, lower_right, count = draw_non_black_rectangle(image, upper_left, lower_right, count, original_image)

    return image, upper_left, lower_right, count


def load_images_from_directory(directory):

    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])

    images = [cv2.imread(file) for file in files]
    return images

def add_text_to_image(image, text):
    font, scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = image.shape[1] - w - 10, image.shape[0] - h - 10
    return cv2.putText(image, text, (x, y), font, scale, color, thickness)

def create_overlay_video(dir, output_video, fps=30):

    images = load_images_from_directory(dir)
    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    upper_left = None
    lower_right = None
    count = 0

    for i, image in tqdm.tqdm(enumerate(images)):
        image, upper_left, lower_right, count = process(image, upper_left, lower_right, count)
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image = add_text_to_image(image, f'{i}')
        video.write(image)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")



dir = '/home/yuming/Documents/mt_data/experimental'
output_video = '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection/output_video_2.mp4'

create_overlay_video(dir, output_video)