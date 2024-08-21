import math
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy import linalg
import numpy.matlib
import random
from scipy.special import erfinv
from scipy import special as sp

pi = math.pi

## =================== RF Parameters ======================================
alpha = 2.5  # Path Loss Exponent
PR = 1
fcRF = 2.1e9
GT = 1
GR = 1
gammaI = (3e8)**2 * GT * GR / (16 * pi**2 * fcRF**2)  # used in SINR 

## ===================== THz Parameters ===========================
PT = 1  # Transmitted Power
kf = 0.05  # Absorption loss
fcTH = 1.0e12
GTT = 316.2
GRR = 316.2

thetabs = pi / 6
thetamt = pi / 6
FBS = thetabs / (2 * pi)
FMT = thetamt / (2 * pi)
prob = FBS * FMT
gammaII = (3e8)**2 * GTT * GRR / (16 * pi**2 * fcTH**2)

R_max = 100
Nit = 10000
lambdas = 10

## ===================== Rate and SINR threshold calculation ========================================
Rate = 5e9 
Wt = 5e8
Wr = 40e6
NP = 10e-10  # Noise Power

## ===================== Aeriation Parameters ========================================
eta_LoS = 1
eta_NLoS = 20  # Corresponds to different environments (Suburban, Urban, etc.)
f_c = 2e9  # Carrier Frequency

# Function to generate an exponential matrix
def generate_exponential_matrix(miu, m, n):
    return np.random.exponential(miu, (m, n))

# Function to sum each row of a matrix
def sum_of_each_row(matrix):
    return matrix.sum(axis=0)

# Function to calculate the RF SINR matrix
def rf_sinr_matrix(distance_matrix):
    NU, NRF = distance_matrix.shape  # Number of vehicles, number of RF base stations
    d_matrix = np.array(distance_matrix)  # Distance matrix [vehicles, bs]
    
    # Generate exponential fading
    fadeRand = generate_exponential_matrix(1, NRF, NU)  # [bs, vehicles]
    
    # Calculate signal matrix for RF
    SRF = gammaI * PR * fadeRand * (d_matrix ** (-alpha)).T
    
    # Calculate interference matrix
    interf = SRF.sum(axis=0) - SRF  # [bs, vehicles]
    
    # Calculate SINR
    SINR = SRF / (NP + interf)
    SNR = SRF / NP
    
    # Calculate data rate
    dr_matrix = Wr * np.log2(SINR + 1)
    
    return dr_matrix.T, interf.T, SINR.T, SNR.T

# Function to calculate the THz SINR matrix
def thz_sinr_matrix(distance_matrix):
    NU, NTHz = distance_matrix.shape  # Number of vehicles, number of THz base stations
    
    d_matrix = np.array(distance_matrix).T
    
    # No fading in THz BSs
    fadeRand1 = generate_exponential_matrix(1, NTHz, NU)  # [bs, vehicles]
    
    # Calculate signal matrix for THz
    STHz = gammaII * fadeRand1 * PT * np.exp(-kf * d_matrix) / (d_matrix**2)
    
    # Calculate interference matrix
    interfT = STHz.sum(axis=0) - STHz
    
    # Calculate SINR
    SINR = STHz / (NP + interfT)
    SNR = STHz / NP
    
    # Calculate data rate
    dr_matrix = Wt * np.log2(SINR + 1)
    
    return dr_matrix.T, interfT.T, SINR.T, SNR.T

# Q-function and inverse Q-function for QoS calculations
def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / np.sqrt(2))

def invQfunc(x):
    return np.sqrt(2) * sp.erfinv(1 - 2 * x)

# QoS calculation functions
def QoS_v(SINR_matrix):
    return 1 - 1 / np.square(1 + SINR_matrix)

L_B = 1
def Qos_epsilon_c(SINR_matrix, W):
    D_t = L_B / W
    b = D_t * SINR_matrix
    V = QoS_v(SINR_matrix)
    epsilon_c = qfunc(np.sqrt(L_B / V) * (np.log2(1 + SINR_matrix) - (np.log(2) * b) / L_B))
    return epsilon_c

def rf_Qos_matrix(SINR_matrix):
    W = Wr
    V = QoS_v(SINR_matrix)
    epsilon_c = Qos_epsilon_c(SINR_matrix, Wr)
    R = (W / np.log(2)) * (np.log2(1 + SINR_matrix) - np.sqrt(V / L_B) * invQfunc(epsilon_c))
    return R.T 

def thz_Qos_matrix(SINR_matrix):
    W = Wt
    V = QoS_v(SINR_matrix)
    epsilon_c = Qos_epsilon_c(SINR_matrix, Wt)
    R = (W / np.log(2)) * (np.log2(1 + SINR_matrix) - np.sqrt(V / L_B) * invQfunc(epsilon_c))

    return R.T

# SINR matrix with threshold calculation
def sinr_with_threshold(sinr_matrix, bs_assignment):
    sinr_matrix_with_threshold = sinr_matrix / (bs_assignment.sum(axis=0) + 1e-8)
    print(sinr_matrix_with_threshold)
    return sinr_matrix_with_threshold

# # Loss calculation function for LoS and NLoS
# def Loss_LoS(dist_2d, i, k):
#     dist_ik = dist_2d[i, k]
#     loss_LoS = 20 * np.log10((4 * pi * f_c * dist_ik) / (3e8 + 1e-7)) + eta_LoS
#     loss_NLoS = 20 * np.log10((4 * pi * f_c * dist_ik) / (3e8 + 1e-7)) + eta_NLoS
#     return loss_LoS, loss_NLoS

# # Angle calculation function
# def theta_ik(dist_3d, vehicles_pos_3d, i, k):
#     dist_3d_ik = dist_3d[i, k]
#     h = vehicles_pos_3d[i, 2]
#     angle = math.atan(h / dist_3d_ik)
#     degrees = math.degrees(angle)
#     return angle, degrees

# # SNR calculation function
# def a2c_link(dist_2d, dist_3d, bs_pos_3d, vehicles_pos_3d):
#     NU, NRF = dist_2d.shape
#     a = 1
#     b = 1
#     A = eta_LoS - eta_NLoS
#     B = 20 * np.log10((4 * pi * f_c) / (3e8 + 1e-7)) + eta_NLoS
    
#     SNR = np.zeros((NU, NRF))
#     for i, row in enumerate(dist_2d):
#         for k, dist_ik in enumerate(row):
#             r_ik = dist_3d[i, k]
#             angleik, thetaik = theta_ik(dist_3d, vehicles_pos_3d, i, k)
#             loss_ik = A / (1 + a * np.exp(-b * ((180 / pi) * angleik) - a)) + \
#                       20 * np.log10(r_ik / math.cos(angleik)) + B
#             SNR[i, k] = PR - loss_ik
    
#     return SNR

