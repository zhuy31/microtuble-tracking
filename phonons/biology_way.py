import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from fresh_implementation import read_tracking_data, fit_spline_equally_spaced, distances, parse_tracking_data_old
import scipy.fft as fft
from functools import partial
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.stats import norm
import  concurrent.futures

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
    
def get_characteristic_time(coeffs, mode_number, time_window = 25, show = True):

    mode = coeffs[:,mode_number]

    y = [abs(acf(mode,i)) for i in range(time_window)]
    x = np.linspace(1,len(y),num=len(y))

    popt, pcov = curve_fit(exp_func, x, y)
    
    if show is True:

        plt.title(r'ACF as a function of $\tau$, mode 0')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'ACF($\tau$)')
        plt.scatter(x,y,s=1)
        fitted_curve = [exp_func(x_i,popt[0],popt[1],popt[2]) for x_i in x]
        plt.scatter(x,fitted_curve,s=1)
        plt.legend([r'ACF data',r'Fitted curve'])
        plt.ylim(min(0,np.min(y)),1+0.1*np.max(y))
        plt.show()

    return 1/popt[1]

def bootstrap(frames, characteristic_time, iterations, mode_number, bins = 50, num_points = 100, show = True, conversion_factor = 1,L_c=1):

    fit_spline_equally_spaced_partial = partial(fit_spline_equally_spaced, num_points=num_points)

    equally_spaced_data = list(map(fit_spline_equally_spaced_partial, frames))
    equally_spaced_data = np.array(equally_spaced_data)

    independent_frames = int(len(frames)/characteristic_time)

    bootstrapped_coeffs = []

    for i in (tqdm(range(iterations)) if show is True else range(iterations)):
        resampled_data = equally_spaced_data[np.random.choice(len(frames),independent_frames, replace=False)]
        dists = distances(resampled_data,angles=True)
        fft_dists = fft.fft(dists,axis=1)
        variances = np.var(fft_dists,axis=0)
        bootstrapped_coeffs.append(variances[mode_number])


    _, bin_edges = np.histogram(bootstrapped_coeffs, bins=bins, density=True)
    _ = (bin_edges[1:] + bin_edges[:-1]) / 2
    mean, std_dev = norm.fit(bootstrapped_coeffs)
    x = np.linspace(min(bootstrapped_coeffs), max(bootstrapped_coeffs), 1000)
    gaussian_curve = norm.pdf(x, mean, std_dev)

    if show is True: 
        plt.hist(bootstrapped_coeffs, bins=bins, density=True)
        plt.plot(x,gaussian_curve)
        plt.show()

    return mean, std_dev

def Lp_curve(n,Lc,N,Lp,e2):
    return (Lc/(n*np.pi))**2 * (1/Lp) + (4/Lc)*e2*(1+(N-1)*((np.sin((n*np.pi)/(2*N)))**2))

def Lp_curve_nonoise(n,Lc,Lp):
    return (Lc/(n*np.pi))**2 * (1/Lp)

def curve_length(points):

    if len(points) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        length += np.sqrt(dx**2 + dy**2)
    
    return length


def gacf(frames,num_points = 100,conversion_factor = 1, L_c = 1):
    fit_spline_equally_spaced_partial = partial(fit_spline_equally_spaced, num_points=num_points)

    equally_spaced_data = list(map(fit_spline_equally_spaced_partial, frames))
    equally_spaced_data = np.array(equally_spaced_data)


    resampled_data = equally_spaced_data
    dists = distances(resampled_data,angles=True)
    fft_dists = fft.fft(dists,axis=1)
    tau = get_characteristic_time(fft_dists,0,100)
    print(f'tau = {tau}')
    return tau

def get_variances(file_path, characteristic_time, iterations,  bins = 30, num_points = 25, bounds = None, conversion_factor = 1e-6):

    frames = read_tracking_data(file_path)

    
    frames = list(frames.values())
    Lc = np.mean(np.array([curve_length(frame) for frame in frames]))
    tau = gacf(frames,conversion_factor=conversion_factor,L_c = Lc)
    characteristic_time = tau
    if bounds is not None:
        frames = frames[bounds[0]:bounds[1]]

    data = [bootstrap(frames, characteristic_time, iterations, i, bins = bins, num_points = num_points, show = False, conversion_factor = conversion_factor, L_c = Lc) for i in tqdm(range(num_points-1))]
    data = data[0:num_points-3]

    y, sigma = list(zip(*data))
    print(sigma)
    x = np.linspace(1,len(y), num = len(y), endpoint= True)
    plt.scatter(x,y,color='Red')

    xfit = np.linspace(1,len(y), num = 1000*len(y), endpoint= True)

    popt1, pcov1 = curve_fit(lambda n, Lp, e2: Lp_curve(n,Lc,num_points,Lp,e2),x,y)
    yfit = Lp_curve(xfit,Lc,num_points,popt1[0],popt1[1])
    plt.scatter(xfit,yfit,color='Blue',s=1)

    popt2, pcov2 = curve_fit(lambda n, Lp: Lp_curve_nonoise(n,Lc,Lp),x[0:4],y[0:4])
    yfit = Lp_curve_nonoise(xfit,Lc,popt2[0])
    plt.scatter(xfit,yfit,color='Green',s=1)
    
    plt.xlim(0,len(x))
    plt.xlabel('Mode number')
    plt.ylabel('Amplitude (Hz)')
    plt.legend(["original curve","fit with noise","fit without noise"], loc="upper right")
    plt.title('Analysis of Fourier Modes')
    plt.show()
    
    print(f"Lp, noise method = {popt1[0]*conversion_factor}")
    print(f"Lp, no noise method = {popt2[0]*conversion_factor}")
    print(f"Lc, pixels = {Lc}")
    return sigma

if __name__ == "__main__":

    file_path_normal= '/home/yuming/Documents/mt_data/mt_data/Python_Snake_Files/Standard/5_31/MT2/MT2.txt'

    #enter file path here!
    
    tubulin_mass = None
    sigma = get_variances(file_path_normal, 15, 5000, bounds=(0,1500), conversion_factor=(0.55))

    file_path_normal= '/home/yuming/Documents/mt_data/mt_data/Python_Snake_Files/HeLa/6_14/MT1/MT1.txt'
    sigma = get_variances(file_path_normal, 15, 5000, bounds=(0,1500), conversion_factor=(0.55))
