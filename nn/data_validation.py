import os
import glob
import cv2
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    return True

def validate_annotations(annotation_file):
    try:
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
        return True
    except FileNotFoundError:
        return False

def validate_dataset(data_dir):
    issues = {
        'missing_images': [],
        'unreadable_images': [],
        'missing_annotations': [],
        'invalid_annotations': []
    }

    # Iterate over each subdirectory in the data directory
    for sub_dir in glob.glob(os.path.join(data_dir, '*')):
        if os.path.isdir(sub_dir):
            image_paths = sorted(glob.glob(os.path.join(sub_dir, '*.jpg')))
            annotation_files = glob.glob(os.path.join(sub_dir, '*.txt'))

            # Check for missing annotation file
            if len(annotation_files) != 1:
                logging.error(f"Expected one annotation file in directory {sub_dir}, but found {len(annotation_files)}.")
                issues['missing_annotations'].append(sub_dir)
                continue

            annotation_file = annotation_files[0]

            # Validate each image
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    logging.warning(f"Missing image file: {image_path}")
                    issues['missing_images'].append(image_path)
                elif not validate_image(image_path):
                    logging.warning(f"Unreadable image file: {image_path}")
                    issues['unreadable_images'].append(image_path)

            # Validate the annotation file
            if not validate_annotations(annotation_file):
                logging.warning(f"Invalid annotation file: {annotation_file}")
                issues['invalid_annotations'].append(annotation_file)

            # Load and validate the content of the annotation file
            annotations = pd.read_csv(annotation_file, delim_whitespace=True, header=None, names=['frame', 'coord_number', 'x', 'y', 'zero'])
            for frame in range(1, len(image_paths) + 1):
                if frame not in annotations['frame'].values:
                    logging.warning(f"No coordinates found for frame: {frame} in {annotation_file}")
                    issues['missing_annotations'].append(f"{annotation_file} (frame {frame})")

    # Log summary of issues
    logging.info(f"Validation completed with the following issues:")
    logging.info(f"Missing images: {len(issues['missing_images'])}")
    logging.info(f"Unreadable images: {len(issues['unreadable_images'])}")
    logging.info(f"Missing annotations: {len(issues['missing_annotations'])}")
    logging.info(f"Invalid annotations: {len(issues['invalid_annotations'])}")

    return issues

if __name__ == "__main__":
    data_dir = "C:/Users/Jackson/Documents/mt_data/postprocessed"
    issues = validate_dataset(data_dir)
    
    # Save issues to a file for further inspection
    with open('dataset_validation_issues.log', 'w') as f:
        for issue_type, paths in issues.items():
            f.write(f"{issue_type}:\n")
            for path in paths:
                f.write(f"  {path}\n")
