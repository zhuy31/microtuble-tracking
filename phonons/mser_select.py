import cv2
import numpy as np
import os

def process(image, n = 10):

    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.bilateralFilter(image, 15, 75, 75) 

    result = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(delta=2, min_area=200, max_area=5000, max_variation=2, min_diversity=0.2)

    regions, _ = mser.detectRegions(gray)
    regions = sorted(regions, key=cv2.contourArea, reverse=True)
    largest_regions = regions[:n]

    # Draw the outlines of the largest regions in red
    for region in largest_regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(result, [hull], 0, (0, 0, 255), 2)  # Red color (BGR format)
    return result


if __name__ == "__main__":
    file_path = '/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm/MT10_30min_200x_1500_138_146pm_t0049.jpg'
    image = process(cv2.imread(file_path))
    cv2.imshow('bomboclart',image)
    cv2.waitKey(0)