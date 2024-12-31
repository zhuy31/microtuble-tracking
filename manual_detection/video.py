import cv2
import os
import cv2.colored_kinfu
import numpy as np
from tqdm import tqdm

def load_images_from_directory(directory):
    # Get list of files and sort them lexicographically
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])
    # Load images
    images = [cv2.imread(file) for file in files]
    return images



def process(image, n = 10):

    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.bilateralFilter(image, 15, 75, 75) 

    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_area = image.shape[0]*image.shape[1]
    mser = cv2.MSER_create(delta=2, min_area=image_area//4, max_area=image_area//2, max_variation=2, min_diversity=0.2)

    regions, _ = mser.detectRegions(gray)
    regions = sorted(regions, key=cv2.contourArea, reverse=True)
    largest_regions = regions[:n]

    # Draw the outlines of the largest regions in red
    for region in largest_regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)  # Red color (BGR format)
    return result


def create_video(dir, output_video, fps=30):

    images = load_images_from_directory(dir)
    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i, image in enumerate(tqdm(images)):
        image = process(image)
        video.write(image)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")


dir = '/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm'
output_video = '/home/yuming/Documents/dev/python/projects/microtuble-tracking/output_video_4.mp4'

create_video(dir, output_video)

