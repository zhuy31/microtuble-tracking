import cv2
import numpy as np

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

def create_overlay_video(frames, output_video, fps=30):
    # Load images from both directories

    
    # Get the size of the images
    height, width  = 500,500
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    image = np.zeros(height, width, 3)
    for frame in frames:
        print(len(frames))
    
    # Release the video writer
    video.release()
    print(f"Video saved as {output_video}")

if __name__ == '__main__':
    file_path_1 = '/home/yuming/Downloads/MT_2/50_per_Hyl_10ms_1000frames_5_MMStack.ome_MT2_cropped-snakes'
    video = '/home/yuming/Documents/dev/python/projects/microtuble-tracking/phonon/video.py'
    create_overlay_video(frames, output_video)