import os
import cv2
import numpy as np

def normalize_image(image):
    return image / 255.0

def resize_image(image, target_size=(512, 512)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def denoise_image(image):
    image = (image * 255).astype(np.uint8)  # Convert to uint8 before denoising
    if len(image.shape) == 2:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, h=3, templateWindowSize=21, searchWindowSize=7)
    else:
        raise ValueError("Unsupported image format for denoising")

def brighten_and_increase_contrast(image, brighten_factor=1.2, contrast_factor=1.2):
    image = image.astype(np.float32)
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brighten_factor * 50)
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
    return adjusted_image

def preprocess_image1(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to remove small noise and fill small holes
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return morph_image

def preprocess_image(image, target_size=(512, 512)):
    image = normalize_image(image)
    image = resize_image(image, target_size)
    image = denoise_image(image)
    image = brighten_and_increase_contrast(image, brighten_factor=1.2, contrast_factor=3)
    image = preprocess_image1(image)
    return image

def save_preprocessed_image(image, filename):
    cv2.imwrite(filename, image)  # Image is already in uint8 format

def main():
    paths = [["image.jpg", "output.jpg"], ["image1.jpg", "output1.jpg"], ["image2.jpg", "output2.jpg"]]

    for path in paths:
        input_path = path[0]
        output_path = path[1]
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            preprocessed_image = preprocess_image(image, target_size=(512, 512))
            save_preprocessed_image(preprocessed_image, output_path)
            print(f"Preprocessed image saved to {output_path}")
        else:
            print(f"Error: Unable to load image {input_path}")

if __name__ == '__main__':
    main()
