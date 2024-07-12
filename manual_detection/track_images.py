import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from tqdm import tqdm
import os
from matplotlib.animation import FuncAnimation


def replace_nonblack_pixels(grayscale_img_path, color_img_path, output_img_path):

    grayscale_img = cv2.imread(grayscale_img_path, cv2.IMREAD_GRAYSCALE)
    color_img = cv2.imread(color_img_path, cv2.IMREAD_COLOR)

    if grayscale_img.shape[:2] != color_img.shape[:2]:
        raise ValueError("The images must have the same dimensions.")

    nonblack_mask = grayscale_img > 0
    

    nonblack_mask = np.stack([nonblack_mask] * 3, axis=-1)
    

    color_img[nonblack_mask] = cv2.merge([grayscale_img] * 3)[nonblack_mask]

    cv2.imwrite(output_img_path, color_img)


def fit_bezier_curve_to_grayscale_image(grayscale_image, control_points_count=6):

    def bezier_curve(t, control_points):
        n = len(control_points) - 1
        curve = np.zeros((len(t), 2))
        for i, cp in enumerate(control_points):
            binom = np.math.comb(n, i)
            curve[:, 0] += binom * (1 - t)**(n - i) * t**i * cp[0]
            curve[:, 1] += binom * (1 - t)**(n - i) * t**i * cp[1]
        return curve

    def mse_loss(control_points, x, y, w):
        control_points = control_points.reshape(-1, 2)
        t = np.linspace(0, 1, len(x))
        curve = bezier_curve(t, control_points)
        dx = curve[:, 0] - x
        dy = curve[:, 1] - y
        return np.sum(w * (dx**2 + dy**2))

    edges = cv2.Canny(grayscale_image, 50, 150)
    y, x = np.nonzero(edges)
    intensities = edges[y, x]

    if len(x) < control_points_count:
        print("Not enough points to fit a Bezier curve.")
        return None, None

    tck, _ = splprep([x, y], s=3)
    u_fine = np.linspace(0, 1, control_points_count)
    x_spline, y_spline = splev(u_fine, tck)

    control_points_initial = np.column_stack((x_spline, y_spline)).flatten()

    # Minimize the weighted MSE
    res = minimize(mse_loss, control_points_initial, args=(x, y, intensities), method='L-BFGS-B')

    if not res.success:
        print(f"Optimization failed: {res.message}")
        return None, None

    control_points_optimized = res.x.reshape(-1, 2)

    u_optimized = np.linspace(0, 1, len(control_points_optimized))

    return control_points_optimized, u_optimized

def bezier_curve(t, control_points):

    n = len(control_points) - 1
    curve = np.zeros((len(t), 2))
    for i, cp in enumerate(control_points):
        binom = np.math.comb(n, i)
        curve[:, 0] += binom * (1 - t)**(n - i) * t**i * cp[0]
        curve[:, 1] += binom * (1 - t)**(n - i) * t**i * cp[1]
    return curve

def save_curve_coordinates(directory, output_file, control_points_count=6, num_points=100):

    files = os.listdir(directory)
    image_files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    
    with open(output_file, 'w') as f:
        frame = 1
        for filename in image_files:
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            control_points, time_scale = fit_bezier_curve_to_grayscale_image(image, control_points_count=control_points_count)
            if control_points is not None:
                t_fine = np.linspace(0, 1, num_points)
                fitted_curve = bezier_curve(t_fine, control_points)
                frame += 1
                for i, (x, y) in enumerate(fitted_curve):
                    f.write(f"{frame}\t{i}\t{x}\t{y}\t0\n")
            else:
                print(f"No Bezier curve found for {filename}")

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

def plot_points_on_image(points, image_shape):

    image = np.zeros(image_shape, dtype=np.uint8)
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), 1, 255, -1)
    return image

import cv2

def save_video_from_coordinates(coordinate_file, image_shape, video_dir, microtubule_dir=None, interval=100, fps=10):
    frames = read_coordinates(coordinate_file)
    images = []
    MSELOSS = [0]
    pastimage = None

    microtubule_files = sorted([f for f in os.listdir(microtubule_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    if len(microtubule_files) < len(frames):
        raise ValueError("Not enough microtubule images for the frames available.")

    for frame_id, microtubule_file in tqdm(zip(sorted(frames.keys()), microtubule_files)):
        points = frames[frame_id]


        microtubule_image = cv2.imread(os.path.join(microtubule_dir, microtubule_file))
        if microtubule_image is None:
            print(f"Error loading image: {microtubule_file}")
            continue
            
        if image_shape is None:
            plot_image = plot_points_on_image(points, image_shape=microtubule_image.shape[:2])
        else:
            plot_image = plot_points_on_image(points, image_shape)

        # Ensure the microtubule image is color
        if len(microtubule_image.shape) == 2 or microtubule_image.shape[2] == 1:
            microtubule_image = cv2.merge([microtubule_image, np.zeros_like(microtubule_image), np.zeros_like(microtubule_image)])
           

        # Ensure the plot image is grayscale
        if len(plot_image.shape) == 3 and plot_image.shape[2] == 3:
            plot_image = cv2.cvtColor(plot_image, cv2.COLOR_BGR2GRAY)
            
        if plot_image.shape != microtubule_image.shape[:2]:
            print(plot_image.shape)
            print(microtubule_image.shape)
            microtubule_image = cv2.resize(microtubule_image, (plot_image.shape[1], plot_image.shape[0]))

        # Convert the grayscale plot image to color
        plot_image_color = cv2.cvtColor(plot_image, cv2.COLOR_GRAY2BGR)

        # Overlay the plot image on the microtubule image
        overlay_image = cv2.addWeighted(microtubule_image, 0.7, plot_image_color, 0.3, 0)

        if pastimage is not None:
            MSELOSS.append((np.sum(plot_image != 0) - np.sum(pastimage != 0))**2)
        pastimage = plot_image
        images.append(overlay_image)

    for i in range(len(MSELOSS) - len(images) + 2):
        MSELOSS.append(0)

    plt.scatter(range(len(MSELOSS)), MSELOSS)
    
    # Define the codec and create VideoWriter object
    video_path = os.path.join(video_dir, 'output_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (microtubule_image.shape[1], microtubule_image.shape[0]), True)

    for image in images:
        video_writer.write(image)

    video_writer.release()

    return MSELOSS

if __name__ == "__main__":
    image_directory = 'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2'  # Change this to the correct directory
    output_file = 'output_coordinates.txt'
    save_curve_coordinates(image_directory, output_file)
    MSELOSS = save_video_from_coordinates(output_file, image_shape=None, video_dir= 'C:/Users/Jackson/Documents/GitHub/microtuble-tracking/manual_detection', microtubule_dir= 'C:/Users/Jackson/Documents/mt_data/experimental',interval=100)

