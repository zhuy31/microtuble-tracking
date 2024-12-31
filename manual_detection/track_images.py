import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import scipy.spatial
from tqdm import tqdm
import os
import math
from matplotlib.animation import FuncAnimation
from skimage.morphology import skeletonize, binary_dilation, square, thin
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from scipy.spatial import distance_matrix

def add_text_to_image(image, text, position='lower_right', margin=10, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, color=(255, 255, 255)):

    height, width = image.shape[:2]

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    if position == 'lower_right':
        x = width - text_width - margin
        y = height - margin
    elif position == 'lower_left':
        x = margin
        y = height - margin
    elif position == 'upper_right':
        x = width - text_width - margin
        y = text_height + margin
    elif position == 'upper_left':
        x = margin
        y = text_height + margin
    else:
        raise ValueError("Position must be one of 'lower_right', 'lower_left', 'upper_right', 'upper_left'")
    
    # Put the text on the image
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    return image


def process_image(image):
    
    _, binary_img = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)
    skeleton = thin(binary_img)
    return skeleton.astype(np.uint16)

def replace_nonblack_pixels(grayscale_img_path, color_img_path, output_img_path):

    grayscale_img = cv2.imread(grayscale_img_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)

    if grayscale_img.shape[:2] != color_img.shape[:2]:
        raise ValueError("The images must have the same dimensions.")

    nonblack_mask = grayscale_img > 0

    nonblack_mask = np.stack([nonblack_mask] * 3, axis=-1)
    

    color_img[nonblack_mask] = cv2.merge([grayscale_img] * 3)[nonblack_mask]

    cv2.imwrite(output_img_path, color_img)


def fit_bezier_curve_to_grayscale_image(grayscale_image, control_points_count=6, num_points = 100):

    grayscale_image = process_image(grayscale_image)
    # Get the coordinates and intensities of all pixels
    y, x = np.indices(grayscale_image.shape)
    intensities = grayscale_image.flatten()
    y = y.flatten()
    x = x.flatten()

    # Filter out black pixels (intensity 0)
    mask = intensities > 0
    x = x[mask]
    y = y[mask]
    
    points = np.column_stack([x, y])
    dist_matrix = distance_matrix(points, points)
    G = nx.from_numpy_array(dist_matrix)
    tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
    
    ordered_x = x[tsp_path]
    ordered_y = y[tsp_path]

    # Fit the parametric spline
    tck, u = splprep([ordered_x, ordered_y  ], s=num_points, k=3)

    # Evaluate the first derivative to approximate arc length
    x_der, y_der = splev(u, tck, der=1)
    ds = np.sqrt(x_der**2 + y_der**2)
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0)  # Insert the starting point for cumulative sum
    s = np.delete(s, 0)

    s_uniform = np.linspace(0, s[-1], num_points + 2)
    s_uniform = np.delete(s_uniform, (0, 1))

    u_uniform = np.interp(s_uniform, s, u)
    x_new, y_new = splev(u_uniform, tck)

    return np.column_stack([x_new,y_new])





def track_image(image_path, control_points_count, num_points, frame):
    # Load image and process it
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fitted_curve = fit_bezier_curve_to_grayscale_image(image, control_points_count=control_points_count, num_points=num_points)
    result = []
    for i, (x, y) in enumerate(fitted_curve):
        result.append(f"{frame}\t{i}\t{x}\t{y}\t0\n")
    return frame, result


def save_curve_coordinates(directory, output_file, control_points_count=6, num_points=400):
    files = os.listdir(directory)
    image_files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    
    frames_results = []
    with ProcessPoolExecutor() as executor, open(output_file, 'w') as f:
        futures = {executor.submit(track_image, os.path.join(directory, filename), control_points_count, num_points, frame): frame 
                   for frame, filename in enumerate(image_files, start=1)}
        
        for future in tqdm(futures):
            frame, result = future.result()
            if result:
                frames_results.append((frame, result))
        
        # Sort results by frame to ensure they are in the correct order
        frames_results.sort(key=lambda x: x[0])
        
        # Write the results in order
        for _, result in frames_results:
            f.writelines(result)

def read_coordinates(file_path):

    frames = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            frame_id = int(parts[0])
            point_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            
            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append((x, y))
    return frames


def curve_length(points):

    if len(points) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        length += math.sqrt(dx**2 + dy**2)
    
    return length

def save_video_from_coordinates(coordinate_file, image_shape, video_dir, microtubule_dir=None, interval=100, fps=10, viewProcessed = False):
    frames = read_coordinates(coordinate_file)
    images = []
    lengths = []

    microtubule_files = sorted([f for f in os.listdir(microtubule_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    if len(microtubule_files) < len(frames):
        raise ValueError("Not enough microtubule images for the frames available.")

    i = 0
    for frame_id, microtubule_file in tqdm(zip(sorted(frames.keys()), microtubule_files)):
        points = frames[frame_id]
        lengths.append(curve_length(points))
        microtubule_image = cv2.imread(os.path.join(microtubule_dir, microtubule_file))
        
        for (x,y) in points:
            y = max(min(int(y+0.5),microtubule_image.shape[0]-1),0)
            x = max(min(int(x+0.5),microtubule_image.shape[1]-1),0)
            microtubule_image[y,x] = [0,0,255]
            
        overlay_image = add_text_to_image(microtubule_image, f'{i}')
        i = i+1
        images.append(overlay_image)

    # Define the codec and create VideoWriter object
    video_path = os.path.join(video_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (microtubule_image.shape[1], microtubule_image.shape[0]), True)

    for image in images:
        video_writer.write(image)

    video_writer.release()

    return lengths

if __name__ == "__main__":
    image_directory ='/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm'
    microtubule_directory = '/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm'
    
    output_file = '/home/yuming/Documents/mt_data/mt_data/HeLa_Snakes/6_13/MT10/MT10_1500'
    #output_file = '/home/yuming/Documents/dev/python/projects/microtubule-tracking-2/rough_coordinates.txt'
    #output_file = '/home/yuming/Documents/dev/python/projects/microtubule-tracking-2/final_coordinates.txt'
    print("saving video...")
    lengths = save_video_from_coordinates(output_file, image_shape=None, fps = 10, video_dir= 'python/projects/microtuble-tracking', 
                                          microtubule_dir= image_directory,interval=100)
    x1 = np.linspace(1,len(lengths),num = len(lengths))
    y1 = lengths
    print(f'variance = {np.var(lengths)}')
    plt.ylim(0,200)
    plt.scatter(x1,y1,s=2, c = "red", label='-1')
    plt.show()
    