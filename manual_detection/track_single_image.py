import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import os
import math
from skimage.morphology import skeletonize, binary_dilation, square

def process_image(image):

    _, binary_img = cv2.threshold(image, 40, 255, cv2.THRESH_BINARY)

    skeleton = skeletonize(binary_img)
    return skeleton.astype(np.uint16)

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

def plot_points_on_image(points, image_shape):
    # Create a blank image with the given shape
    image = np.zeros(image_shape, dtype=np.uint8)
    
    # Plot each point on the image
    for (x, y) in points:
        image[int(y+0.5), int(x+0.5)] = 255 

    return image

def process_single_image(image_path, output_path=None):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    control_points, _ = fit_bezier_curve_to_grayscale_image(image)
    if control_points is None:
        print("Failed to fit a Bezier curve.")
        return

    t_fine = np.linspace(0, 1, 400)
    fitted_curve = bezier_curve(t_fine, control_points)

    plot_image = plot_points_on_image(fitted_curve, image.shape)
    plot_image_color = cv2.merge([ np.zeros_like(plot_image), np.zeros_like(plot_image), plot_image])

    microtubule_image = cv2.imread(image_path)
    if microtubule_image.shape[:2] != plot_image.shape:
        microtubule_image = cv2.resize(microtubule_image, (plot_image.shape[1], plot_image.shape[0]))

    overlay_image = cv2.addWeighted(microtubule_image.astype(np.uint8), 0.5, plot_image_color.astype(np.uint8), 3, 0)
    
    if output_path:
        cv2.imwrite(output_path, overlay_image)
    

if __name__ == "__main__":
    image_path = 'C:/Users/Jackson/Documents/mt_data/experimental2/image_269.png'  # Change this to the correct path
    output_path = 'overlayed_image.jpg'  # Optional: specify an output path to save the image
    process_single_image(image_path, output_path)
