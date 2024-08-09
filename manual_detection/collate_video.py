import cv2
import os
import numpy as np

def load_images_from_directory(directory):
    # Get list of files and sort them lexicographically
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])
    # Load images
    images = [cv2.imread(file) for file in files]
    return images

def create_overlay_video(dir1, dir2, output_video, fps=30):
    # Load images from both directories
    images1 = load_images_from_directory(dir1)
    images2 = load_images_from_directory(dir2)
    
    # Ensure both directories have the same number of images
    if len(images1) != len(images2):
        print("The directories must contain the same number of images.")
        return
    
    # Get the size of the images
    height, width, _ = images1[0].shape
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Overlay images and write to video
    for img1, img2 in zip(images1, images2):
        # Resize images to the same size if necessary
        img2_resized = cv2.resize(img2, (width, height))
        
        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
        img1 = cv2.merge([np.zeros_like(img1),np.zeros_like(img1),img1])
        overlay_img = cv2.addWeighted(img1, 0.5, img2_resized, 1, 0)
        
        # Write the overlay image to the video
        video.write(overlay_img)
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")

# Example usage
dir1 = '/home/yuming/Documents/mt_data/preprocessed/imageset2'
dir2 = '/home/yuming/Documents/mt_data/experimental'
output_video = '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection/output_video_1.mp4'

create_overlay_video(dir1, dir2, output_video)
