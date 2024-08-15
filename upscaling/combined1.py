import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import fast_tsp
from skimage.morphology import thin
from fil_finder import FilFinder2D
import astropy.units as u
import pandas as pd
from scipy.interpolate import interp1d
from io import BytesIO
from multiprocessing import Pool
from functools import partial


def load_images_from_directory(directory):

    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])

    images = [cv2.imread(file) for file in files]
    return images

def add_text_to_image(image, text):
    font, scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = image.shape[1] - w - 10, image.shape[0] - h - 10
    return cv2.putText(image, text, (x, y), font, scale, color, thickness)

def process_single_2(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return image.astype(np.uint8)

def process_single(image):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18,18))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    image = cv2.medianBlur(image,9)
    return image

def process_stack(stack, window, h = 20):
    denoised_image = cv2.fastNlMeansDenoisingColoredMulti(stack, window, window*2+1, None, h, 10, 7, 21)
    return denoised_image

def create_overlay_video(dir, output_video, fps=10, window = 3):

    images = load_images_from_directory(dir)
    height, width, _ = images[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    #single-image processing
    with Pool(12) as p:
        images = p.map(process_single, images)
    
    
    #create inputs
    inputs = []
    for i in tqdm.tqdm(range(len(images))):
        if i >= window and i < len(images)-window:
            inputs.append(images[i-window:i+window+1]) 

    #local temporal (e.g. multi-image) processing
    denoised_images = []
    with Pool(12) as p:
        process_stack_partial = partial(process_stack, window = window)
        denoised_images = p.map(process_stack_partial,inputs[-30:])

    print("done")

    with Pool(12) as p:
        denoised_images = p.map(process_single_2,denoised_images)

    plt.imshow(denoised_images[-1])
    plt.show()

    for i, image in tqdm.tqdm(enumerate(denoised_images)):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = add_text_to_image(image, f'{i}')
        video.write(image)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")




dir = '/home/yuming/Documents/mt_data/experimental'
output_video = '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection/output_video_2.mp4'
output_dir = '/home/yuming/Documents/mt_data/preprocessed/imageset3'

create_overlay_video(dir,output_video)