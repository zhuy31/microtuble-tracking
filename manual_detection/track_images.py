import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from tqdm import tqdm
import os
import math
from matplotlib.animation import FuncAnimation
from skimage.morphology import skeletonize, binary_dilation, square

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


    skeleton = skeletonize(binary_img)
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


def fit_bezier_curve_to_grayscale_image(grayscale_image, control_points_count=6):

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
    intensities = intensities[mask]

    if len(x) < control_points_count:
        print("Not enough points to fit a Bezier curve.")
        return None, None

    # Fit a spline to the detected edge points
    tck, _ = splprep([x, y])
    u_fine = np.linspace(0, 1, control_points_count)
    x_spline, y_spline = splev(u_fine, tck)

    # Use the spline points as the initial guess for the control points
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
        binom = math.comb(n, i)
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

def save_curve_coordinates(directory, output_file, control_points_count=6, num_points=400):

    files = os.listdir(directory)
    image_files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    
    with open(output_file, 'w') as f:
        frame = 1
        for filename in tqdm(image_files):
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
    import numpy as np
    
    # Create a blank image with the given shape
    image = np.zeros(image_shape, dtype=np.uint8)
    
    # Plot each point on the image
    for (x, y) in points:
        #if int(y*image_shape[0]/256) < image_shape[0] and int(x*image_shape[1]/256) < image_shape[1]: 
        #    image[int(y*image_shape[0]/256), int(x*image_shape[1]/256)] = 255 
        image[int(y+0.5), int(x+0.5)] = 255 

    return image

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
        if microtubule_image is None:
            print(f"Error loading image: {microtubule_file}")
            continue
            
        if image_shape is None:
            plot_image = plot_points_on_image(points, image_shape=microtubule_image.shape[:2])
        else:
            plot_image = plot_points_on_image(points, image_shape)

        # Ensure the microtubule image is color
        if len(microtubule_image.shape) == 2 or microtubule_image.shape[2] == 1:
            microtubule_image = cv2.cvtColor(microtubule_image, cv2.COLOR_GRAY2RGB)
           

        # Ensure the plot image is grayscale
        if len(plot_image.shape) == 3 and plot_image.shape[2] == 3:
            plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGB2GRAY)
            
        if plot_image.shape != microtubule_image.shape[:2]:
            print(plot_image.shape)
            print(microtubule_image.shape)
            microtubule_image = cv2.resize(microtubule_image, (plot_image.shape[1], plot_image.shape[0]))

        # Convert the grayscale plot image to color

        plot_image_color = cv2.merge([ np.zeros_like(plot_image), np.zeros_like(plot_image), plot_image])

        if viewProcessed is True:
            microtubule_image = process_image(cv2.cvtColor(microtubule_image, cv2.COLOR_RGB2GRAY))
            microtubule_image = (microtubule_image - microtubule_image.min()) / (microtubule_image.max() - microtubule_image.min()) * 255
            microtubule_image = cv2.merge([ np.zeros_like(microtubule_image), microtubule_image,  np.zeros_like(microtubule_image)])
            
        else:
            temp = process_image(cv2.cvtColor(microtubule_image, cv2.COLOR_RGB2GRAY))
            temp = (temp - temp.min()) / (temp.max() - temp.min()) * 255
            temp = cv2.cvtColor(temp.astype(np.uint8), cv2.COLOR_GRAY2RGB)  
            cv2.imwrite(f'C:/Users/Jackson/Documents/mt_data/experimental2/image_{i}.png', temp)


        overlay_image = cv2.addWeighted(microtubule_image.astype(np.uint8), 0.5, plot_image_color.astype(np.uint8), 0.7, 0)
        overlay_image = add_text_to_image(overlay_image, f'{i}')
        i = i+1

        pastimage = plot_image

        if viewProcessed is True:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            overlay_image = cv2.filter2D(overlay_image, -1, kernel)
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
    image_directory = 'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2'  # Change this to the correct directory
    output_file = 'output_coordinates.txt'
    print("tracking curves...")
    save_curve_coordinates(image_directory, output_file)
    print("saving video...")
    lengths = save_video_from_coordinates(output_file, image_shape=None, fps = 10, video_dir= 'C:/Users/Jackson/Documents/GitHub/microtuble-tracking/manual_detection', microtubule_dir= 'C:/Users/Jackson/Documents/mt_data/preprocessed/imageset2',interval=100, viewProcessed=True)
    x = np.linspace(1,len(lengths),num = len(lengths))
    y = lengths
    plt.scatter(x,y)
    plt.show()
    