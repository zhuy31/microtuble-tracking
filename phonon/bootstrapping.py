import numpy as np
from fresh_implementation import read_tracking_data, fit_spline_equally_spaced, distances, parse_tracking_data_old
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm
import scipy.fft as fft
import matplotlib.pyplot as plt
from scipy.stats import norm

def bootstrap(file_path, iterations, independent_frames, mode_number, num_points = 100, show = True):
        #load in data
    frames = read_tracking_data(file_path)
    frames = list(frames.values())

    #partially evaluate function to tell it number of points
    fit_spline_equally_spaced_partial = partial(fit_spline_equally_spaced,num_points=num_points)

    #spline-fit and then segment into needed number of segements
    equally_spaced_data = list(map(fit_spline_equally_spaced_partial,frames))
    equally_spaced_data, arclengths  = [e[0] for e in equally_spaced_data], [e[1] for e in equally_spaced_data]
    equally_spaced_data = np.array(equally_spaced_data)


    bootstrapped_coeffs = []

    for i in tqdm(range(iterations)):
        resampled_data = equally_spaced_data[np.random.choice(len(frames),independent_frames, replace=False)]
        dists = distances(resampled_data)
        fft_dists = fft.fft(dists,axis=1)
        variances = np.var(fft_dists,axis=0)
        bootstrapped_coeffs.append(variances[mode_number])

    data = bootstrapped_coeffs
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    if show is True:    # Plot the histogram.
        plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        plt.show()
    return mu

if __name__ == "__main__":
    file_path = '/home/yuming/Documents/mt_data/MT9/MT9.txt'
    bootstrap(file_path, 5000, 200, 0, show=True)