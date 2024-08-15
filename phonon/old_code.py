import numpy as np
import scipy.constants as const

# Import data
txt1 = np.loadtxt('/home/yuming/Downloads/XYBeadsCoordNB26NT998.txt')
txt2 = txt1[:, 1:]  # txt2 doesn't have frame num
frame_num = int(txt1[-1, 0])
txt2 = txt2.T
rows, cols = txt2.shape

# Convert 2D matrix to 3D
three_dim = np.zeros((rows, 2, frame_num))
x_increment = 0
y_increment = 1

for z in range(frame_num):  # loop to convert to 3D matrix
    three_dim[:, 0, z] = txt2[:, x_increment]
    three_dim[:, 1, z] = txt2[:, y_increment]
    x_increment += 2
    y_increment += 2

coord_reconstructed = three_dim

# Constants
k_b = const.Boltzmann  # Units = m^2 kg s^-2 K^-1
T = 293  # Units = Kelvin

# Mass Matrix
dimer_MW = 110  # kilograms/mol
dimer_mass = dimer_MW / const.Avogadro  # kilograms/molecule
turn_mass = 13 * dimer_mass
mass_per_pixel = (60 / 7.4) * turn_mass  # kg

# Average microtubule length in pixels and meters
num_beads = rows

def mt_length(frame_num, coord_reconstructed):
    lengths = np.zeros(frame_num)
    for i in range(frame_num):
        diff = np.diff(coord_reconstructed[:, :, i], axis=0)
        lengths[i] = np.sum(np.linalg.norm(diff, axis=1))
    return np.mean(lengths), np.mean(lengths) * (1e-6 / 15.3)

L_avg, L_avg_meters = mt_length(frame_num, coord_reconstructed)
mass_per_segment = (L_avg / (num_beads - 1)) * mass_per_pixel  # kg

# Linear density mass/length
density = dimer_mass * 1625 * 1e6  # mass per meter
mass_per_segment = density * (L_avg / (num_beads - 1))  # kg

# Calculate magnitude of displacement
mag = np.zeros((num_beads, frame_num))

for frame in range(frame_num):
    for row in range(num_beads):
        mag[row, frame] = np.linalg.norm(coord_reconstructed[row, :, frame])

mag_meters = mag * (1e-6 / 15.3)  # units = meters

# Calculate displacement
disp = np.zeros_like(mag_meters)

for frame in range(frame_num):
    disp[:, frame] = mag_meters[:, frame] - np.mean(mag_meters, axis=1)

# Fourier transform the displacement matrix over beads
U_q = np.fft.fft(disp, axis=0)  # units = meters

# Variance of the coefficients
variance_fft = np.var(U_q, axis=1)  # units = m^2

# Getting F(q)
F_q = (2 * k_b * T) / variance_fft  # units

# Get frequencies
w_q = np.sqrt(F_q / dimer_mass)
freq_q = w_q / (2 * np.pi)
freq_q_kHz = freq_q / 1000

# Old way of getting k vector
n = np.arange(-13, 14)
N = (num_beads - 1) / 2
a = L_avg_meters / (num_beads - 1)
k_vector = (n * np.pi) / (N * a)
k_a = k_vector * a

# KMatrix

# Generate an inverse matrix for the exponents
EXP = np.zeros((num_beads, num_beads), dtype=complex)

for bead in range(num_beads):
    for q in range(num_beads):
        EXP[bead, q] = np.exp((1j * 2 * np.pi * q * bead) / num_beads)

EXP_inv = np.linalg.inv(EXP)

# Generate K matrix
K_n = np.dot(EXP_inv, F_q)
K_n_real = np.real(K_n)
K_n_magnitude = np.abs(K_n)
K_n_real_diag = np.diag(K_n_real)
K_n_mag_diag = np.diag(K_n_magnitude)

print(K_n)