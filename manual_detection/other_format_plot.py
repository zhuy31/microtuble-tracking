import numpy as np
import math
import matplotlib.pyplot as plt

def read_coordinates(file_path):

    frames = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
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
    points = sorted(points, key=lambda coord: (coord[1], -coord[0]))
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
    file_path = '/home/yuming/Documents/dev/python/microtuble-tracking/manual_detection/output_coordinates.txt'  # Change this to the correct path
    lengths = calculate_lengths(file_path)
    x = []
    y = []
    for frame, length in lengths.items():
        x.append(frame)
        y.append(length)
    plt.scatter(x,y)
    plt.ylim(0, 150)
    plt.show()

