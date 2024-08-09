import cv2
import numpy as np

def upscale_image(image_path, scale_factor=4):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return image, upscaled_image

def display_images_side_by_side(image1, image2,screen_width=1920, screen_height=1080):
    combined_image = np.hstack((cv2.resize(image1,(image2.shape[1],image2.shape[0])), image2))

    # Get the screen resolution
    screen_aspect_ratio = screen_width / screen_height
    combined_aspect_ratio = combined_image.shape[1] / combined_image.shape[0]

    if combined_aspect_ratio > screen_aspect_ratio:
        new_width = screen_width
        new_height = int(screen_width / combined_aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(screen_height * combined_aspect_ratio)

    resized_image = cv2.resize(combined_image, (new_width, new_height))

    # Display the resized image
    cv2.imshow('Original and 4x Upscaled Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = '/home/yuming/Documents/mt_data/experimental/MT10_30min_200x_1500_138_146pm_t1494.jpg'

original_image, upscaled_image = upscale_image(image_path, scale_factor=4)
display_images_side_by_side(original_image, upscaled_image)
