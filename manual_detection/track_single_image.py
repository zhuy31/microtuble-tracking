import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import thinning

def find_contours(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return contour

def track_image(image):
    _, image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
    image = thinning.guo_hall_thinning(image)
    contour = find_contours(image)
    contour = contour.squeeze()
    x = contour[:, 0]
    y = contour[:, 1]

    # Calculate the cumulative distance (arc length) along the contour
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.concatenate([[0], np.cumsum(distances)])
    
    # Normalize t to range [0, 1]
    t = t / t[-1]
    tck_x = interpolate.splrep(t, x, s=15)
    tck_y = interpolate.splrep(t, y, s=15)
    t_new = np.linspace(0, 1, 400)

    # Evaluate the spline fits for x and y
    x_new = interpolate.splev(t_new, tck_x, der=0)
    y_new = interpolate.splev(t_new, tck_y, der=0)
    return x_new, y_new



if __name__ == "__main__":
    image_path = '/home/yuming/Documents/mt_data/preprocessed/imageset2/MT10_30min_200x_1500_138_146pm_t0269.jpg'  # Replace with your image path
    track_image(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE))