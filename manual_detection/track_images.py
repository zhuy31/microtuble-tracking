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
from scipy import interpolate
from fil_finder import FilFinder2D
from astropy.io import fits
from astropy import units as u
import pandas as pd
from skimage.morphology import thin
import warnings

warnings.filterwarnings(
    action="ignore",
    message="Graph pruning reached max iterations.",
)
warnings.filterwarnings(
    action="ignore",
    message="Using inputted mask. Skipping creation of anew mask."
)

def process_image(image):
    return image

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


def extract_longest_branch(skeleton):

    fil = FilFinder2D(skeleton, distance=250 * u.pc, beamwidth= 1 * u.pix,mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    for idx, filament in enumerate(fil.filaments): 

        data = filament.branch_properties.copy()
        data_df = pd.DataFrame(data)
        data_df['offset_pixels'] = data_df['pixels'].apply(lambda x: x+filament.pixel_extents[0])

        longest_branch_idx = data_df.length.idxmax()
        longest_branch_pix = data_df.offset_pixels.iloc[longest_branch_idx]
        y,x = longest_branch_pix[:,0],longest_branch_pix[:,1]

    output_image = np.zeros_like(skeleton)

    for x2,x1 in zip(y,x):
        x2 = min(max(int(x2),0),output_image.shape[0]-1)
        x1 = min(max(int(x1),0),output_image.shape[1]-1)
        output_image[x2,x1] = 255

    return output_image


def connected_components(image, threshold=0):

    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned_image = np.zeros_like(image)

    max_area = -1
    max_pointer = None
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max_area:
            max_pointer = i
            max_area = stats[i, cv2.CC_STAT_AREA]

    cleaned_image[labels == max_pointer] = image[labels == max_pointer]
    return cleaned_image

def find_contours(image):
    _, thresh = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    return contour

    
def process(image):

    image = cv2.filter2D(image,-1,np.ones((8,8),np.float32)/64)
    

    _, image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
    image = thin(image).astype(np.uint8)
    image = connected_components(image)

    image = extract_longest_branch(image)

    return image

def track_image(image):

    image = process(image)
    contour = find_contours(image)
    contour = contour.squeeze()
    x = contour[:, 0]
    y = contour[:, 1]

    # Calculate the cumulative distance (arc length) along the contour
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    t = np.concatenate([[0], np.cumsum(distances)])
    
    # Normalize t to range [0, 1]
    t = t / t[-1]
    tck_x = interpolate.splrep(t, x, s=15)
    tck_y = interpolate.splrep(t, y, s=15)
    t_new = np.linspace(0, 1, 400)

    # Evaluate the spline fits for x and y
    x_new = interpolate.splev(t_new, tck_x, der=0)
    y_new = interpolate.splev(t_new, tck_y, der=0)
    return zip(x_new, y_new)

def save_curve_coordinates(directory, output_file, control_points_count=6, num_points=400, no_bezier_fit = False):

    files = os.listdir(directory)
    image_files = sorted([f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'))])
    
    with open(output_file, 'w') as f:
        frame = 1
        if no_bezier_fit is False:
            for filename in tqdm(image_files):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                fitted_curve = track_image(image)
                
                frame += 1
                for i, (x, y) in enumerate(fitted_curve):
                    f.write(f"{frame}\t{i}\t{x}\t{y}\t0\n")
        else:
            for filename in tqdm(image_files):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = process_image(image)
                frame += 1
                nonblack_points = np.transpose(np.nonzero(image))
                
                for i, (x, y) in enumerate(nonblack_points):

                    f.write(f"{frame}\t{i}\t{y}\t{x}\t0\n")

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
    for (x, y) in points:
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
    image_directory = '/home/yuming/Documents/mt_data/mt_data/preprocessed/imageset2'  # Change this to the correct directory
    output_file = '/home/yuming/Documents/dev/python/projects/microtuble-tracking/output_coordinates.txt'
    save_curve_coordinates(image_directory, output_file)
    print("saving video...")
    lengths = save_video_from_coordinates(output_file, image_shape=None, fps = 10, video_dir= '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection',
                                           microtubule_dir= '/home/yuming/Documents/mt_data/mt_data/preprocessed/imageset2',interval=100, viewProcessed=True)
    x = np.linspace(1,len(lengths),num = len(lengths))
    y = lengths
    plt.scatter(x,y)
    plt.ylim(0,250)
    plt.show()
    