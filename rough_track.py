from tqdm import tqdm
import numpy as np
import os
import cv2
import concurrent.futures
from scipy.interpolate import splprep, splev
import networkx as nx
from scipy.spatial import distance_matrix

def display_images_with_bbox(images):
    image = images[0]
    images = images[::int(len(images)/10)]
    # Select ROI 
    r = cv2.selectROI("select the area", image) 
    cropped_images = np.hstack(np.array([process(image[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])],past_regions=None)[0] for image in images])) 

    cv2.imshow("Cropped and processed images", cropped_images) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return (int(r[1]), int(r[1]+r[3]), int(r[0]), int(r[0]+r[2]))

def load_images_from_directory(directory):
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('png', 'jpg', 'jpeg'))])
    print("loading images...")
    images = [cv2.imread(file) for file in tqdm(files)]
    return images

def sum_intersections(region, past_regions):
        region_hull = cv2.convexHull(np.array(region))
        return sum(cv2.intersectConvexConvex(region_hull, cv2.convexHull(np.array(r)))[0]/cv2.contourArea(cv2.convexHull(np.array(r))) for r in past_regions)   

def denoise_and_crop(image,bbox):
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.fastNlMeansDenoisingColored(image,None,25,15,7,21)
    image = cv2.bilateralFilter(image, 15, 75, 75) 
    image = image[bbox[0]:bbox[1],bbox[2]:bbox[3]]
    return image

def denoise_and_crop_images(images, bbox):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(lambda image:denoise_and_crop(image, bbox), x) for x in images]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results, [image[bbox[0]:bbox[1],bbox[2]:bbox[3]] for image in images]

def process(image, original_image = None, n = 10, past_regions = None):

    if original_image is None:
        result = image.copy()
    else:
        result = original_image

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_area = image.shape[0]*image.shape[1]
    mser = cv2.MSER_create(delta=2, min_area=image_area//64, max_area=image_area//8, max_variation=1, min_diversity=0.1)

    regions, _ = mser.detectRegions(gray)

    if past_regions is not None:
        regions = sorted(regions, key=lambda region : sum_intersections(region, past_regions), reverse=True)
    else:
        regions = sorted(regions, key=cv2.contourArea, reverse=True)

    if len(regions) == 0:
         regions = past_regions

    hull = cv2.convexHull(regions[0].reshape(-1, 1, 2))
    cv2.drawContours(result, [hull], 0, (0, 0, 255), 1)

    aux_image = np.zeros_like(gray)
    cv2.fillPoly(aux_image, [regions[0]], 255)
    coords = track_image(aux_image)
    for coord in coords:
        result[int(coord[0]),int(coord[1])]=[255,0,0]

    return result, regions[:n], coords

def track_image(aux_image):
    skeleton = cv2.ximgproc.thinning(cv2.threshold(aux_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    coordinates = np.argwhere(skeleton == 255)
    return coordinates

        
    
def segment_video(images, output_video, rough_coordinates_file = None):
    bbox = display_images_with_bbox(images)
    print("denoising images...")
    f = open(rough_coordinates_file, "w")
    f.write(f"bbox_shape {bbox}")
    f.close()
    images, original_images = denoise_and_crop_images(images, bbox)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 30, (images[0].shape[1],images[0].shape[0]))

    regions = None
    print("processing and writing to video...")
    with open(rough_coordinates_file, 'a') as file:
        for i, image in enumerate(tqdm(images)):
            image, regions, coords = process(image, original_image = original_images[i], past_regions=regions)
            video.write(image)
            for j, (x, y) in enumerate(coords):
                file.write(f"{i}\t{j}\t{x}\t{y}\t0\n")
    
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")



# Example usage
if __name__ == "__main__":
    
    image_dir = '/home/yuming/Documents/mt_data/MT10_30min_200x_1500_138_146pm'
    output_video = 'python/projects/microtuble-tracking/output_video.mp4'
    rough_coordinates_file = '/home/yuming/Documents/dev/python/projects/microtubule-tracking-2/rough_coordinates.txt'
    images = load_images_from_directory(image_dir)
    segment_video(images, output_video, rough_coordinates_file = rough_coordinates_file)
