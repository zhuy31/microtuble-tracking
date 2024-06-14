import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
import cv2
import numpy as np

def keep_largest_bounding_box_component(image):
    assert image.ndim == 2, "Input image should be a binary (2D) image"
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    if num_labels <= 1:
        return image

    # Find the component with the largest bounding box
    largest_bbox_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT])
    
    largest_component = np.zeros_like(image)
    largest_component[labels == largest_bbox_label] = 255
    
    return largest_component


def normalize_image(image):
    return image / 255.0

def resize_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def denoise_image(image):
    image = (image * 255).astype(np.uint8)  # Convert to uint8 before denoising
    if len(image.shape) == 2:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 1)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        raise ValueError("Unsupported image format for denoising")

def brighten_image(image, factor=3):
    """
    Brightens the image by multiplying the pixel values by the given factor.
    Values are clipped to the range [0, 255] to prevent overflow.
    """
    brightened_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return brightened_image

def binarize_image(image, threshold=128):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def preprocess_image(image, target_size=(512, 512), binarize_threshold=128, brighten_factor=1.2,kernel_size=5, iterations=5):
    image = normalize_image(image)
    image = resize_image(image, target_size)
    image = apply_gaussian_blur(image)
    image = denoise_image(image)
    image = brighten_image(image, brighten_factor)
    image = cv2.dilate(image,np.ones(shape=(20,20)),iterations = 1)
    image = binarize_image(image, binarize_threshold)    
    skeleton = skeletonize(image)
    image = (skeleton * 255).astype(np.uint8)
    return image

def save_preprocessed_image(image, filename):
    cv2.imwrite(filename, image)  # Image is already in uint8 format

def main():
    input_path = 'image.jpg'
    output_path = 'outputimage.png'
    binarize_threshold =  48 # Adjust this value to set the threshold for binarization
    brighten_factor = 3 # Adjust this value to set the brightness factor

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        preprocessed_image = preprocess_image(image, target_size=(512, 512), binarize_threshold=binarize_threshold, brighten_factor=brighten_factor)
        save_preprocessed_image(preprocessed_image, output_path)
        print(f"Preprocessed image saved to {output_path}")
    else:
        print(f"Error: Unable to load image {input_path}")

if __name__ == '__main__':
    main()
