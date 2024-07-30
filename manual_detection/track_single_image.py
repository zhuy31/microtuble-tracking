import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import cv2
import os
from scipy.interpolate import UnivariateSpline
def get_nonzero_entries(image):
    """
    Returns a list of all nonzero entries in the given image.
    
    Parameters:
    image (numpy.ndarray): The input image.
    
    Returns:
    list: A list of tuples, each containing the coordinates and the value of a nonzero entry.
    """
    nonzero_coords = np.nonzero(image)
    nonzero_values = image[nonzero_coords]
    nonzero_entries = list(zip(nonzero_coords[0], nonzero_coords[1], nonzero_values))
    nonzero_entries = [x[:2] for x in nonzero_entries]
    return list(set(nonzero_entries))


def interpolate_pixelated_line(points, smoothing_factor=30):
    # Ensure the points are sorted by x or y
    points = sorted(points, key=lambda p: p[0])

    # Extract x and y coordinates from points
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Parameterize the points by their cumulative distance
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    # Fit smoothing splines to x and y as functions of the parameter
    spline_x = UnivariateSpline(cumulative_distances, x, s=smoothing_factor)
    spline_y = UnivariateSpline(cumulative_distances, y, s=smoothing_factor)

    # Generate new points using the parameterization
    u_new = np.linspace(0, cumulative_distances[-1], 1000)
    x_new = spline_x(u_new)
    y_new = spline_y(u_new)

    return x_new, y_new
def get_nth_image(directory, n):
    return cv2.imread(sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[n])

pixelated_points = get_nonzero_entries(get_nth_image('C:/Users/Jackson/Documents/mt_data/experimental2',337))
# Interpolate the pixelated line
x_smooth, y_smooth = interpolate_pixelated_line(pixelated_points)

# Plot the original pixelated line and the smooth interpolated line
plt.imshow(get_nth_image('C:/Users/Jackson/Documents/mt_data/experimental2',500))
plt.show()
