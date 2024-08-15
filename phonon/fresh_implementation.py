import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import cv2
#load in tracking data from OLD tracking .txt format (neither the current imageJ plugin nor my python code)
def parse_tracking_data_old(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    
    for i in range(0, len(lines), 2):
        x_coords = list(map(float, lines[i].split()[1:]))
        y_coords = list(map(float, lines[i + 1].split()[1:]))
        coords = np.array(list(zip(x_coords, y_coords)))
        coordinates.append(coords)

    return np.array(coordinates)

#find distance array, that is, convert (x,y) -> L2 distance from mean
def distances(coords):
    frames = coords.shape[0]

    #this is a (Nx2) matrix, N = number of points used to track, representing mean positions
    means = np.mean(coords,axis=0)

    #distance array
    dists = coords-np.array(np.array(frames*[means]))
    dists = np.linalg.norm(coords, 2, axis=2)
    
    return dists

#calculate FFT of data, take variations, find F(q), and then omega(q)
def fourier_variations(dists, k_b = 1.38e-23, T = 298, m = 1e-10):
    fft_dists = fft.fft(dists,axis=1)
    variances = np.var(fft_dists,axis=0)
    F = 2*k_b*T/variances
    omegas = np.sqrt(F/m)

    #convert from interval [0,2pi] to [-pi,pi]
    omegas = np.concatenate((np.array_split(omegas,2)[1],np.array_split(omegas,2)[0]))

    return omegas



if __name__ == '__main__':
    file_path = '/home/yuming/Downloads/MT_1/txt1_27beads.txt'
    coordinates_array = parse_tracking_data_old(file_path)
    dists= distances(coordinates_array)
    omegas = fourier_variations(dists)

    #plot data
    plt.scatter(np.linspace(0,2*np.pi,num=len(omegas)),omegas)
    plt.show()
