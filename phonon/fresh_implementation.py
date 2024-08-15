import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

def parse_tracking_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    
    for i in range(0, len(lines), 2):
        x_coords = list(map(float, lines[i].split()[1:]))
        y_coords = list(map(float, lines[i + 1].split()[1:]))
        coords = np.array(list(zip(x_coords, y_coords)))
        coordinates.append(coords)

    return np.array(coordinates)

def distances(coords):
    frames = coords.shape[0]
    means = np.mean(coords,axis=0)
    dists = coords-np.array(np.array(frames*[means]))
    dists = np.linalg.norm(coords, 2, axis=2)
    
    return dists

def fourier_variations(dists, k_b = 1.38e-23, T = 298, m = 100):
    dists = dists
    fft_dists = fft.fft(dists,axis=0)
    variances = np.var(fft_dists,axis=0)
    F = 2*k_b*T/variances
    omegas = np.sqrt(F/m)
    return omegas

if __name__ == '__main__':
    file_path = '/home/yuming/Downloads/20180521_50_per_Hyl_10ms_1000frames_123_concatenated/MT_1/txt1_27beads.txt'
    coordinates_array = parse_tracking_data(file_path)
    dists= distances(coordinates_array)
    omegas = fourier_variations(dists)
    plt.scatter(np.linspace(-np.pi,np.pi,num=len(omegas)),omegas)
    plt.show()
