import os
import numpy as np
import cv2
import imgaug.augmenters as iaa

def normalize_image(image):
    return image / 255.0

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

# Image augmentation sequence (Updated to use Resize instead of Scale)
aug = iaa.Sequential([
    iaa.Rotate((-10, 10)),
    iaa.Resize((0.8, 1.2)),
    iaa.TranslateX(percent=(-0.1, 0.1)), 
    iaa.TranslateY(percent=(-0.1, 0.1)),
    iaa.Flipud(0.5),
    iaa.Fliplr(0.5),
    iaa.ElasticTransformation(alpha=50, sigma=5)
])

def denoise_image(image):
    image = (image * 255).astype(np.uint8)  # Convert to uint8 before denoising
    if len(image.shape) == 2:  # Grayscale image
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        raise ValueError("Unsupported image format for denoising")

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def remove_small_objects(image, min_size=150):
    # Convert image to binary
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create an output image to hold the cleaned image
    cleaned_image = np.zeros_like(image)
    
    # Loop through all connected components
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = image[labels == i]
    
    return cleaned_image

def preprocess_image(image, target_size):
    # Normalize
    image = normalize_image(image)
    
    # Resize
    image = resize_image(image, target_size)
    
    # Denoise
    image = denoise_image(image)
    
    # Enhance contrast
    if len(image.shape) == 2:  # Grayscale image
        image = enhance_contrast(image)
    
    # Remove small objects
    image = remove_small_objects(image)
    
    return image

def save_preprocessed_image(image, filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, filename)
    cv2.imwrite(output_path, image)  # Image is already in uint8 format
    print(f"Saved preprocessed image to {output_path}")

# Example usage
def main():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'microtubule_image.jpg')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        preprocessed_image = preprocess_image(image, (512, 512))
        save_preprocessed_image(preprocessed_image, 'preprocessed_microtubule_image.png')
    else:
        print(f"Error: Unable to load image at {image_path}")

if __name__ == '__main__':
    main()