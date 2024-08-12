import cv2 
import numpy as np
import matplotlib.pyplot as plt

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

    return image, upper_left, lower_right, count

def process2(image, corner1, corner2, target_size = None):
    x1, y1 = corner1
    x2, y2 = corner2

    

    cropped_image = image[y1:y2, x1:x2]

    cropped_image = cv2.normalize(cropped_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image, None, 30, 10, 7, 21)
    if len(cropped_image.shape)==3:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped_image = clahe.apply(cropped_image)
    cropped_image = cv2.normalize(cropped_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    cropped_image = get_largest_connected_component(cropped_image)
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image[y1:y2, x1:x2] = cropped_image

    return image

image_path = "/home/yuming/Documents/mt_data/experimental/MT10_30min_200x_1500_138_146pm_t1188.jpg"
plt.imshow(process2(process(image_path),(0,0),()))
plt.show()

