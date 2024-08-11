import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

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

def draw_non_black_rectangle(image, factor = 1.5):

    non_black_pixels = np.where(image > 0)
    
    if non_black_pixels[0].size > 0:

        upper_left = (np.min(non_black_pixels[1]), np.min(non_black_pixels[0]))
        lower_right = (np.max(non_black_pixels[1]), np.max(non_black_pixels[0]))
        upper_left, lower_right = resize(upper_left, lower_right, image, factor)
        image = cv2.rectangle(image, upper_left, lower_right, (255, 0, 0), 1)

    return image

def get_largest_connected_component(image):


    num_labels, labels = cv2.connectedComponents(image)
    
    if num_labels <= 1:
        return np.zeros_like(image)
    
    largest_label = 1 + np.argmax(np.bincount(labels.flat)[1:])
    largest_component = np.zeros_like(image)
    largest_component[labels == largest_label] = 255
    
    return largest_component

def process(image):
    image = cv2.normalize(
    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    original_image = image
    image = cv2.fastNlMeansDenoisingColored(image, None,15,10,7,21)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    _, image = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
    image = cv2.erode(image, np.ones((4,4)), iterations=1) 
    image = cv2.dilate(image, np.ones((24,24)), iterations=1) 
    image = get_largest_connected_component(image)

    image = draw_non_black_rectangle(image)

    colored = cv2.merge([np.zeros_like(image),np.zeros_like(image),image])


    return cv2.addWeighted(colored,0.5,original_image,0.5,0)

image_path = "/home/yuming/Documents/mt_data/experimental/MT10_30min_200x_1500_138_146pm_t1497.jpg"
plt.imshow(process(cv2.imread(image_path)))
plt.show()

def load_images_from_directory(directory):

    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])

    images = [cv2.imread(file) for file in files]
    return images

def create_overlay_video(dir, output_video, fps=30):

    images = load_images_from_directory(dir)
    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Overlay images and write to video
    for image in tqdm.tqdm(images):
        video.write(process(image))
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")



dir = '/home/yuming/Documents/mt_data/experimental'
output_video = '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection/output_video_2.mp4'

create_overlay_video(dir, output_video)
