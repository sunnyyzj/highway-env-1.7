
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
alpha =2.5                   #%Path Loss Exponent better to have indoor small
PR = 1
fcRF=2.1e9
GT=1
GR=1
gammaI=(3e8)**2 * GT * GR/(16 * pi ** 2 * fcRF **2);  # %used in sinr 


## ===================== THz Parameters ===========================
PT = 1                          #   % Tranmitted Power
kf = 0.05                        #   % Absorbtion loss
fcTH=1.0e12
GTT=316.2
GRR=316.2
# GTT=31.62;
# GRR=31.62

thetabs=pi/6;#%%%in degrees
thetamt=pi/6;#%%%in degrees
FBS=thetabs/(2*pi)#;
FMT=thetamt/(2*pi)#;
prob= FBS*FMT#;
gammaII=(3e8)**2 * GTT * GRR /(16 * pi ** 2 * fcTH ** 2)#;
 

R_max = 100#;
Nit = 10000#;
lambdas=10#;

## ===================== Rate and SINR theshold calculation ========================================
arr_num = [0] * 5
# print(arr_num)

Rate = 5e9 
Wt=5e8
Wr=40e6
NP = 10e-10
#SINRthRF =2.^(Rate./(Wr))-1 #%thresholds
#SINRthTH =2.^(Rate./(Wt))-1

#Bias=[1000 100 1 0.001 0.0001 0.0001 0.00001]#%%%0.05
#%Bias=[ 10^6  10^5 10^4  10^4 10^3 10^3 10^3];%%%0.2

## ===================== Aeriation Parameters ========================================
eta_LoS = 1
eta_NLoS = 20 # (0.1, 21), (1.0,20), (1.6, 23), (2.3, 34) corresponding to Suburban, Urban,Dense Urban, and Highrise Urban respectively
f_c = 2e9 # Carrier Frequency is 850MHz 8.5e+8


def generate_exponential_matrix(miu,m,n):
    """
    Generate Exponential matrix with mean miu with m rows and n columns
    ref https://www.geeksforgeeks.org/numpy-random-exponential-in-python/
    """
    return np.random.exponential(miu, (m, n))
    # exponential_matrix = []
    # for i in range(m):
    #     row = np.random.exponential(miu, n)
    #     exponential_matrix.append(row)
    # return exponential_matrix

def sum_of_each_row(matrix):
    # arr = []
    # for i in range(len(matrix)):
    #     arr.append(np.sum(matrix[i]))
    # # arr = arr.transpose()
    # return arr
    return matrix.sum(axis=0)



def rf_sinr_matrix(distance_matrix):
    """
    Convert distance matrix to sinr matrix

    """
    NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs
    # print('NU,NRF\n',NU,NRF)
    d_matrix = np.array(distance_matrix)    # [v, bs]
    # print('d_matrix\n',d_matrix)
    fadeRand = generate_exponential_matrix(1,NRF,NU)    # [bs, v]
    # fadeRand = generate_exponential_matrix(1,NU,NRF)    # [bs, v]

    # print('fadeRand\n',fadeRand)
    #signal matrix for RF
    # SRF = gammaI*fadeRand*PR*d_matrix
    SRF = gammaI * PR * fadeRand * (d_matrix ** (-1 * alpha)).T
    # print('gammaI\n',gammaI)
    # SRF = SRF ** (-1 * alpha) # [bs, v]

    interf = SRF.sum(axis=0) - SRF # [bs, v]
    # print('SRF\n',SRF)
    # print('SRF.sum(axis=0)\n',SRF.sum(axis=0))
    # SRF.sum(axis=0)
    # print('interf\n',interf)
    
    NP=10e-10 #(10) ** (-10)
    RPrAllu1 = Wr * np.log2(SRF / (NP + interf) + 1) #.T # [v, bs]
    # interf = interf.T # [v, bs]
    SINR = SRF / (NP + interf)
    SNR = SRF / (NP)
    dr_matrix = RPrAllu1
    interf_matrix = interf
    # print('dr_matrix\n',dr_matrix)

    # print(dr_matrix.shape)
    # print(interf_matrix.shape)
    # print(SINR.shape)
    # print(SNR.shape)


    return dr_matrix,interf_matrix,SINR,SNR


def thz_sinr_matrix(distance_matrix):
    # """
    # Convert distance matrix to sinr matrix

    # """
    # NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs

    # d_matrix = np.array(distance_matrix)    # [v, bs]

    # fadeRand = generate_exponential_matrix(1,NRF,NU)    # [bs, v]

    # #signal matrix for RF
    # # SRF = gammaI*fadeRand*PR*d_matrix
    # SRF = gammaI * PR * fadeRand * d_matrix.T

    # SRF = SRF ** (-1 * alpha) # [bs, v]

    # interf = SRF.sum(axis=0) - SRF # [bs, v]
    
    # NP=10e-10 #(10) ** (-10)
    # RPrAllu1 = Wr * np.log2(SRF / (NP + interf) + 1).T # [v, bs]
    # interf = interf.T # [v, bs]

    # sinr_matrix = RPrAllu1
    # interf_matrix = interf
    # # print(df)

    # return sinr_matrix,interf_matrix
    """
    Convert distance matrix to sinr matrix

    """
    
    NU,NTHz = distance_matrix.shape # row is vehicle, column is rf bs

    d_matrix = np.array(distance_matrix)
    # print('d_matrix\n',d_matrix)
    fadeRand1 = generate_exponential_matrix(1,NTHz,NU)
    # print('fadeRand1\n',fadeRand1)

    d_matrix = d_matrix.T

    # SRF = PT * gammaII * fadeRand * d_matrix.T
    # SRF = SRF ** (-1*alpha) #fractional_matrix_power(distance_matrix_thz,2)
    # interf = SRF.sum(axis=0) - SRF
    # NP=10e-10 #(10) ** (-10)
    # RPrAllu1 = Wr * np.log2(1 + SRF / (NP + interf)).T

    STHz = gammaII * fadeRand1 * PT * np.exp(-kf * d_matrix) / (d_matrix**2) # signal matrix for THZ
    # print('STHz\n',STHz)
    interfT = STHz.sum(axis=0) - STHz 
    # interfT = np.tile(np.sum(STHz, axis=0), (NTHz, 1)) - STHz # interference matrix for THz
    # print('interfT\n',interfT)
    TPrAllu1 = Wt * np.log2(1 + STHz / (NP + interfT))

    interf = interfT#.T

    dr_matrix = TPrAllu1
    interf_matrix = interf

    SINR = STHz / (NP + interf)
    SNR = STHz / (NP)
    # print('TPrAllu1\n',TPrAllu1)
    # print('TSINR\n',SINR)
    # print('TSNR\n',SNR)

    # print(dr_matrix.shape)
    # print(interf_matrix.shape)
    # print(SINR.shape)
    # print(SNR.shape)

    return dr_matrix,interf_matrix,SINR,SNR

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def invQfunc(x):
    return np.sqrt(2)*sp.erfinv(1-2*x)

def QoS_v(SINR_matrix):
    return 1 - 1/(np.square(1+SINR_matrix))


L_B = 10
def Qos_epsilon_c(SINR_matrix,W):
    D_t = L_B/W
    # W = PR PT
    b = D_t * SINR_matrix
    V = QoS_v(SINR_matrix)
    # epsilon_c = qfunc(np.sqrt(D_t*W/V) *(np.log(1 + SINR_matrix) - (np.log(2) * b )/(D_t * W) ))
    epsilon_c = qfunc(np.sqrt(L_B/V) *(np.log(1 + SINR_matrix) - (np.log(2) * b )/(L_B) ))
    # epsilon_c.where(epsilon_c >= 0.8, 0.8, epsilon_c)
    epsilon_c = np.where(epsilon_c >= 0.8, 0.8, epsilon_c)
    return epsilon_c

def rf_Qos_matrix(SINR_matrix):
    # NU,NRF = rf_dr_matrix.shape # row is vehicle, column is rf bs
    # print('sinr_rf\n',SINR_matrix)
    W = Wr
    V = QoS_v(SINR_matrix)
    # print('V_rf\n',V)
    epsilon_c = Qos_epsilon_c(SINR_matrix,Wr)
    # print('epsilon_c_rf\n',epsilon_c)
    R = ( W / np.log(2)) * (np.log2(1 + SINR_matrix) - np.sqrt(V / L_B) * invQfunc(epsilon_c) )
    # print('qos_dr_rf\n',R)

    return R 

def thz_Qos_matrix(SINR_matrix):
    # NU,NTHz = thz_dr_matrix.shape # row is vehicle, column is THz bs
    # print('sinr_thz\n',SINR_matrix)
    W = Wt
    V = QoS_v(SINR_matrix)
    # print('V_thz\n',V)
    epsilon_c = Qos_epsilon_c(SINR_matrix,Wt)
    # print('epsilon_c_thz\n',epsilon_c)
    R = ( W / np.log(2)) * (np.log(1 + SINR_matrix) - np.sqrt(V / L_B) * invQfunc(epsilon_c) )
    # print('qos_dr_thz\n',R)
    return R 



def sinr_with_threshold(sinr_matrix, bs_assignment):
    ''' 
    Input 
    1.sinr matrix (merged rf thz version)
    2.bs assignment matrix
    we devide the sinr matrix to the bss
                            rf1    rf2    rf3    rf4    rf5   rf6    rf7    th1    th2    th3    th4    th5    th6
    sinr                    10     20     30     20     20    40     10     20     30     40     40     30     20
    bss assignment          4      8      9      9      2     10     3      3      3      2      4      1      3
    data with threshold	    2.5   2.5     10/3   20/9   10    4      10/3   20/3   10     20     10     30     20/3
    Output
    sinr matrix with threshold
    '''
    sinr_matrix_with_threshold = sinr_matrix / (bs_assignment.sum(axis = 0) + 1e-8)
    # sinr_matrix_with_threshold = sinr_matrix.div(bs_assignment.sum(),index=sinr_matrix.columns)

    return sinr_matrix_with_threshold
