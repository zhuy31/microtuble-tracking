import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fresh_implementation import read_tracking_data, fit_spline_equally_spaced, distances, parse_tracking_data_old
import scipy.fft as fft
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm

def acf(coeffs, tau):

    N = coeffs.shape[0]
    

    mean = np.mean(coeffs)
    
    num_arr = np.array([(coeffs[t] - mean) * np.conjugate(coeffs[t + tau] - mean) for t in range(0, N - tau)])
    num = np.sum(num_arr)
    
    denom = np.sum(np.abs(coeffs - mean) ** 2)    

    result = num / denom
    return result

def exp_func(x, a, b, c):
    return a*np.exp(-b*x+c)

def get_fourier_coeffs(file_path, num_points = 100):

    dists = distances(parse_tracking_data_old(file_path))
    dists = dists[0:250,:]

    coeffs = fft.fft(dists,axis=1)

    return coeffs
    

def get_characteristic_time(coeffs, mode_number, time_window = 25, show = True):

    mode = coeffs[:,mode_number]

    y = [abs(acf(mode,i)) for i in range(time_window)]
    x = np.linspace(1,len(y),num=len(y))

    popt, pcov = curve_fit(exp_func, x, y)
    
    if show is True:
        plt.scatter(x,y)
        fitted_curve = [exp_func(x_i,popt[0],popt[1],popt[2]) for x_i in x]
        plt.scatter(x,fitted_curve)
        plt.ylim(min(0,np.min(y)),1+0.1*np.max(y))
        plt.show()

    return 1/popt[1]

def bootstrap(file_path, characteristic_time, iterations, mode_number, num_points = 100):
    frames = parse_tracking_data_old(file_path)
    equally_spaced_data = frames

    independent_frames = int(len(frames)/characteristic_time)

    bootstrapped_coeffs = []

    for i in tqdm(range(iterations)):
        resampled_data = equally_spaced_data[np.random.choice(len(frames),independent_frames, replace=False)]
        dists = distances(resampled_data)
        fft_dists = fft.fft(dists,axis=1)
        variances = np.var(fft_dists,axis=0)
        bootstrapped_coeffs.append(variances[mode_number])

    plt.hist(bootstrapped_coeffs, bins = 200)
    plt.show()

if __name__ == "__main__":
    # Input parameters
    file_path = '/home/yuming/Downloads/MT_1/50_per_Hyl_10ms_1000frames_5_MMStack.ome_MT1_cropped-snakes'
    file_path_1 = '/home/yuming/Downloads/MT_1/txt1_27beads.txt'
    fourier_coeffs = get_fourier_coeffs(file_path_1)

    
    tau_c = get_characteristic_time(fourier_coeffs, 1, show=True)
    print(tau_c)
    

    

