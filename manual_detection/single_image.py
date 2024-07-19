import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from tqdm import tqdm
import os
import math
from matplotlib.animation import FuncAnimation
from skimage.morphology import skeletonize, binary_dilation, square


def process_image(image):

    # Threshold the image
    _, binary_img = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)

    skeleton = binary_img.astype(np.uint16)
    skeleton = (skeleton - skeleton.min()) / (skeleton.max() - skeleton.min()) * 255
    return skeleton

def display_nth_image(directory, n, process_image):
    try:
        # Get a list of all files in the directory
        files = sorted(os.listdir(directory))
        
        # Filter out non-image files (optional, based on common image extensions)
        image_files = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'))]
        
        if n < 1 or n > len(image_files):
            print(f"Error: Please provide a number between 1 and {len(image_files)}")
            return
        
        # Get the nth image file
        nth_image_file = image_files[n - 1]
        
        # Construct the full file path
        image_path = os.path.join(directory, nth_image_file)
        
        # Open the image
        with Image.open(image_path) as img:
            # Convert the image to grayscale
            gray_img = img.convert("L")
            
            # Convert the PIL image to a numpy array
            img_array = np.array(gray_img)
            
            # Process the grayscale image using the provided function
            processed_img_array = process_image(img_array)
            
            # Convert the processed numpy array back to a PIL image
            processed_img = Image.fromarray(processed_img_array)
            
            # Display the processed image
            processed_img.show()
    
    except Exception as e:
        print(f"An error occurred: {e}")


display_nth_image('C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2',1,process_image=process_image)