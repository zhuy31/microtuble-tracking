import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def create_gaussian(x, y, height, width, sigma=1):
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
    return gaussian

def detect_center_line(image, sigma=1, smooth_sigma=5):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    height, width = gray_image.shape
    gaussian_sum = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            if gray_image[y, x] > 0:
                gaussian = create_gaussian(x, y, height, width, sigma)
                gaussian_sum += gaussian
    
    central_line_indices = np.argmax(gaussian_sum, axis=0)
    
    # Mask for valid indices
    mask = gray_image[central_line_indices, np.arange(width)] > 0
    valid_indices = np.where(mask, central_line_indices, np.nan)
    
    # Find the first and last valid indices
    valid_range = np.where(~np.isnan(valid_indices))[0]
    if len(valid_range) > 0:
        start, end = valid_range[0], valid_range[-1]
        # Interpolate missing values within the valid range
        indices = np.arange(start, end + 1)
        valid = ~np.isnan(valid_indices[indices])
        interpolated_indices = np.interp(indices, indices[valid], valid_indices[indices][valid])
        
        # Smooth the line using Gaussian filter
        smooth_indices = gaussian_filter1d(interpolated_indices, smooth_sigma)
        central_line_indices[start:end + 1] = smooth_indices.astype(int)
    
    # Create output image
    output_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for x in range(width):
        if start <= x <= end:
            output_image[central_line_indices[x], x] = [0, 0, 255]
    
    return output_image

def main():
    sigma = 2
    smooth_sigma = 5  # Standard deviation for the Gaussian smoothing filter

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'preprocessed_microtubule_image.png')
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    output_image = detect_center_line(image, sigma, smooth_sigma)
    
    output_image_path = os.path.join(script_dir, 'central_line_detected.png')
    cv2.imwrite(output_image_path, output_image)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('Central Line Detected')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()