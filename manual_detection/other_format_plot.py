import numpy as np
import math
import matplotlib.pyplot as plt

def read_coordinates(file_path):

    frames = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                frame_id = int(parts[0])
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

def calculate_lengths(file_path):

    frames = read_coordinates(file_path)
    lengths = {}
    for frame_id, points in frames.items():
        lengths[frame_id] = curve_length(points)
    return lengths



if __name__ == "__main__":
    file_path = '/home/yuming/Documents/mt_data/mt_data/HeLa_Snakes/6_13/MT10/MT10_1500'
    lengths = calculate_lengths(file_path)
    x = []
    y = []
    for frame, length in lengths.items():
        x.append(frame)
        y.append(length)
    plt.scatter(x,y,s=1,label='0',color='blue',alpha=0.5)
    
    file_path = '/home/yuming/Documents/dev/python/projects/microtubule-tracking-2/final_coordinates.txt'
    lengths = calculate_lengths(file_path)
    x = []
    y = []
    for frame, length in lengths.items():
        x.append(frame)
        y.append(length)
    plt.scatter(x,y,s=1,label='1',color='red',alpha=0.5)
    plt.legend(['ImageJ','Rough-Fine Method'])
    plt.xlabel('Frame')
    plt.ylabel('Length')
    plt.title('Comparision of lengths v. frames ')
    plt.show()

