import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import thinning
from fil_finder import FilFinder2D
import astropy.units as u
from skimage.morphology import thin
from plantcv import plantcv as pcv
import numpy as np
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
from astropy.io import fits
from astropy import units as u
import pandas as pd

def extract_longest_branch(skeleton):

    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
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
        output_image[int(x2),int(x1)] = 255

    
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
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image = cv2.erode(image, np.ones((8, 8), np.uint8) , cv2.BORDER_REFLECT)  
    image = cv2.filter2D(image,-1,np.ones((8,8),np.float32)/64)

    _, image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
    image = thin(image).astype(np.uint8)
    image = connected_components(image)
    image = extract_longest_branch(image)

    return image

def track_image(image_path):
    image = process(cv2.imread(image_path,cv2.IMREAD_GRAYSCALE))
    print(image.shape)
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

def overlay_points(image_path, points):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    for x, y in points:
        plt.plot(x, y, 'ro')
    plt.axis('off')
    plt.show()

def find_contours(image):

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    
    return contour

if __name__ == "__main__":
    image_path = '/home/yuming/Documents/mt_data/preprocessed/imageset2/MT10_30min_200x_1500_138_146pm_t1450.jpg'  # Replace with your image path

    pts = track_image(image_path)
    overlay_points(image_path,pts)