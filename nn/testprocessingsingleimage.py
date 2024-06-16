import cv2
import numpy as np

def fill_black_with_noise(image):
    mask = image == 0
    noise = np.random.normal(127, 127, image.shape).astype(np.uint8)
    image[mask] = noise[mask]
    return image

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle_rad = np.deg2rad(angle)
    cos = np.abs(np.cos(angle_rad))
    sin = np.abs(np.sin(angle_rad))
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rotated_image = fill_black_with_noise(rotated_image)
    scale_x = 512 / new_w
    scale_y = 512 / new_h
    resized_image = cv2.resize(rotated_image, (512, 512), interpolation=cv2.INTER_AREA)
    return resized_image, M, scale_x, scale_y

def transform_coordinates(coords, M, scale_x, scale_y):
    transformed_coords = coords.copy()
    for i in range(len(coords)):
        x, y = coords[i, 2], coords[i, 3]
        # Apply the rotation matrix to the coordinates
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        # Apply the scaling to the coordinates
        new_x *= scale_x
        new_y *= scale_y
        transformed_coords[i, 2], transformed_coords[i, 3] = new_x, new_y
    return transformed_coords

def process_single_image(image_path, txt_path, angle):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return
    
    coords = np.loadtxt(txt_path)
    if coords.size == 0:
        print(f"Error: Unable to read coordinates from {txt_path}")
        return
    
    coords = coords[coords[:, 0] == 2]  # Only use the coordinates for the second image
    
    rotated_image, M, scale_x, scale_y = rotate_image(image, angle)
    transformed_coords = transform_coordinates(coords, M, scale_x, scale_y)
    
    # Overlay the transformed coordinates on the image
    for coord in transformed_coords:
        x, y = int(coord[2]), int(coord[3])
        cv2.circle(rotated_image, (x, y), 3, (255, 0, 0), -1)
    
    # Display the image with coordinates overlayed
    cv2.imshow('Transformed Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the correct paths for the image and text file
    image_path = 'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset1/MT7_30min_100x_443_453pm_1500_t0001.jpg'
    txt_path = 'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset1/rescaled_coords.txt'
    angle = 45  # Specify the rotation angle

    process_single_image(image_path, txt_path, angle)
