import numpy as np
import math

###############################################################################
#  1.  Constants / helpers
###############################################################################
c0 = 3e8                            # speed of light (m s‑1)
pi = math.pi

# ---------- RF & THz shared parameters ----------
Wr = 40e6                           # RF bandwidth (Hz)
Wt = 500e6                          # THz bandwidth (Hz)

k_BOLTZ = 1.38064852e-23            # Boltzmann’s constant
T_noise  = 290                      # receiver noise temperature (K)
N0_dBm   = 10*np.log10(k_BOLTZ*T_noise*1000)  # ≈ –174 dBm/Hz
NP_dBm   = N0_dBm + 10*np.log10(Wr) # total noise power in dBm

# ---------- Air–to–ground (A2G) link parameters ----------
alpha_a2g = 2.1                     # path‑loss exponent (LoS)
eta_LoS   = 1                       # excess loss (dB)
eta_NLoS  = 20
f_c       = 2.1e9                   # carrier (Hz)

# Logistic‑fit constants from 3GPP TR 36.777 (urban macro)
a_env, b_env = 9.61, 0.16           # environment‑dependent
                                    # (≈ –1.5 / 3.5 in your draft)

# Transmit / noise powers in linear scale (W)
P_TX_W    = 10**(40/10) * 1e-3      # 40 dBm
P_NOISE_W = 10**(NP_dBm/10) * 1e-3

def pl_free_space(dist, f_hz):
    """Free‑space path loss (linear gain)  – LoS dominant."""
    λ = c0 / f_hz
    return (4*pi*dist/λ)**2         # *linear* scale

###############################################################################
#  2.  Vectorised A2C routine
###############################################################################
def a2c_link(dist_2d, dist_3d, veh_pos_3d):
    """
    Air‑to‑cellular link capacity (bit/s) for every UAV-BS pair.

    Parameters
    ----------
    dist_2d :  (N_uav, N_bs) array – horizontal separation (m)
    dist_3d :  (N_uav, N_bs) array - 3‑D distance  (m)
    veh_pos_3d : (N_uav, 3) array - UAV positions (x,y,h)

    Returns
    -------
    R : (N_uav, N_bs) array – achievable rate (bit/s)
    """
    # --------------- Elevation angle & LoS‑probability -----------------
    h = veh_pos_3d[:, 2:3]                          # (N_uav,1) for broadcasting
    θ_rad = np.arctan2(h, dist_2d)                  # same shape as dist_2d
    θ_deg = np.degrees(θ_rad)
    P_LoS = 1 / (1 + a_env*np.exp(-b_env*(θ_deg - a_env)))  # 3GPP logistic

    # --------------- Path‑loss (linear scale) --------------------------
    # Separate LoS / NLoS to include different η
    PL_LoS = pl_free_space(dist_3d, f_c) * 10**(eta_LoS/10)
    PL_NLoS= pl_free_space(dist_3d, f_c) * 10**(eta_NLoS/10)
    PL = P_LoS*PL_LoS + (1-P_LoS)*PL_NLoS          # mixture

    # --------------- Received power & interference ---------------------
    # Small‑scale fading ~ Nakagami‑m (m=1 → Rayleigh).  Sample once.
    m_fade = 1
    H = np.random.gamma(m_fade, 1/m_fade, PL.shape)

    P_rx = P_TX_W * H / PL                         # (N_uav,N_bs)

    # Aggregate interference seen by each UAV ( ∑ over BSs )
    I = P_rx.sum(axis=1, keepdims=True) - P_rx     # broadcast subtraction

    # --------------- SIR & rate per link -------------------------------
    SIR = P_rx / (I + P_NOISE_W)                  # linear
    R   = Wr * np.log2(1 + SIR)

    return R

###############################################################################
#  HAPS data-rate matrix
###############################################################################

###############################################################################
#  0.  Physical & noise constants – reuse / share with a2c_link
###############################################################################
c0       = 3e8                     # speed of light (m s‑1)
pi       = math.pi
k_BOLTZ  = 1.38064852e-23          # Boltzmann (J K‑1)
T_noise  = 290                     # (K)
N0_dBmHz = 10*np.log10(k_BOLTZ*T_noise*1000)  # −174 dBm/Hz baseline

###############################################################################
#  1.  UAV → HAPS data‑rate matrix
###############################################################################
def haps_datarate_matrix(dist_haps,
                         b_ratio,
                         p_ratio,
                         B_max      = 20e6,     # total system BW   (Hz)
                         P_max_dBm  = 30,       # UAV TX power cap  (dBm)
                         fc         = 2e9,      # carrier           (Hz)
                         G_tx_rx    = 20,       # combined antenna gain (dBi)
                         K_rice     = 10,       # Rice factor (linear)
                         seed       = None):
    """
    Achievable throughput per UAV on orthogonal UAV‑HAPS links.

    Parameters
    ----------
    dist_haps : (N_uav, N_haps) array
        Slant distance(s) UAV ↔︎ HAPS (m).
    b_ratio   : (N_uav, N_haps) array
        Fraction of `B_max` allocated to each link (Σ≤1 across UAVs).
    p_ratio   : (N_uav, N_haps) array
        Fraction of `P_max_dBm` allocated to each link (≤1 per UAV).
    B_max     : float
        System bandwidth available to HAPS (Hz).
    P_max_dBm : float
        Maximum per‑UAV transmit power (dBm).
    fc        : float
        Carrier frequency (Hz).
    G_tx_rx   : float
        *Combined* TX‑RX directional antenna gain (dBi).
    K_rice    : float
        Rice K‑factor for small‑scale fading (linear).  K→∞ ⇒ deterministic LoS.
    seed      : int | None
        RNG seed for reproducibility.

    Returns
    -------
    R : (N_uav, N_haps) ndarray
        Achievable data rate (bit s‑1) on each orthogonal link.
    """
    if seed is not None:
        np.random.seed(seed)

    # --------------- Sanity shapes -----------------
    dist_haps = np.atleast_2d(dist_haps)
    b_ratio   = np.atleast_2d(b_ratio)
    p_ratio   = np.atleast_2d(p_ratio)
    assert dist_haps.shape == b_ratio.shape == p_ratio.shape, \
           "dist_haps, b_ratio, p_ratio must share the same shape."

    # --------------- Link budget -------------------
    λ      = c0 / fc
    G_lin  = 10**(G_tx_rx/10)                         # antenna gain (linear)
    FSPL   = (4*pi*dist_haps/λ)**2                   # free‑space path‑loss
    beta   = G_lin / FSPL                            # large‑scale gain

    # small‑scale Ricean fading |h|²  (unit mean)
    sigma = 1 / np.sqrt(2*(K_rice+1))
    h_LOS = np.sqrt(K_rice/(K_rice+1))
    fading = (h_LOS + sigma*np.random.randn(*dist_haps.shape))**2 + \
             (sigma*np.random.randn(*dist_haps.shape))**2

    G_ch = beta * fading                             # |h|² * path‑loss * gain

    # --------------- Power & noise -----------------
    P_max_W = 10**(P_max_dBm/10) * 1e-3              # dBm → W
    P_tx    = P_max_W * p_ratio                      # allocated TX power (W)

    N0_WHz  = 10**(N0_dBmHz/10) * 1e-3               # W/Hz
    B_link  = B_max * b_ratio                        # per‑link bandwidth
    Np      = N0_WHz * B_link                        # integrated noise (W)

    # --------------- SNR & rate --------------------
    SNR  = P_tx * G_ch / Np                          # linear
    R    = B_link * np.log2(1 + SNR)                 # bit/s

    return R
# import math
# import numpy as np
# import pandas as pd
# from scipy.linalg import fractional_matrix_power
# from scipy import linalg
# import numpy.matlib
# import random
# import warnings

# pi = math.pi

# ## =================== RF Parameters ======================================
# alpha =2.5                   #%Path Loss Exponent better to have indoor small
# PR = 1
# fcRF=2.1e9
# GT=1
# GR=1
# gammaI=(3e8)**2 * GT * GR/(16 * pi ** 2 * fcRF **2);  # %used in sinr 


# ## ===================== THz Parameters ===========================
# PT = 1                          #   % Tranmitted Power
# kf = 0.05                        #   % Absorbtion loss
# fcTH=1.0e12
# GTT=316.2
# GRR=316.2
# # GTT=31.62;
# # GRR=31.62

# thetabs=pi/6;#%%%in degrees
# thetamt=pi/6;#%%%in degrees
# FBS=thetabs/(2*pi)#;
# FMT=thetamt/(2*pi)#;
# prob= FBS*FMT#;
# gammaII=(3e8)**2 * GTT * GRR /(16 * pi ** 2 * fcTH ** 2)#;
 

# R_max = 100#;
# Nit = 10000#;
# lambdas=10#;

# ## ===================== Rate and SINR theshold calculation ========================================
# arr_num = [0] * 5
# # print(arr_num)

# Rate = 5e9 
# Wt=5e8
# Wr=40e6
# NP = 10e-10
# #SINRthRF =2.^(Rate./(Wr))-1 #%thresholds
# #SINRthTH =2.^(Rate./(Wt))-1

# #Bias=[1000 100 1 0.001 0.0001 0.0001 0.00001]#%%%0.05
# #%Bias=[ 10^6  10^5 10^4  10^4 10^3 10^3 10^3];%%%0.2




# def generate_exponential_matrix(miu,m,n):
#     """
#     Generate Exponential matrix with mean miu with m rows and n columns
#     ref https://www.geeksforgeeks.org/numpy-random-exponential-in-python/
#     """
#     return np.random.exponential(miu, (m, n))
#     # exponential_matrix = []
#     # for i in range(m):
#     #     row = np.random.exponential(miu, n)
#     #     exponential_matrix.append(row)
#     # return exponential_matrix

# def sum_of_each_row(matrix):
#     # arr = []
#     # for i in range(len(matrix)):
#     #     arr.append(np.sum(matrix[i]))
#     # # arr = arr.transpose()
#     # return arr
#     return matrix.sum(axis=0)

# def rf_sinr_matrix__raw(distance_matrix,vehicles,bss):
#     """
#     Convert distance matrix to sinr matrix

#     """
#     NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs
#     # print(distance_matrix)
#     # print("distance_matrix.shape",NU,NRF )
#     # NU = len(distance_matrix) 
#     # NRF = len(distance_matrix[0])

#     d_matrix = np.array(distance_matrix)    # [v, bs]

#     fadeRand = generate_exponential_matrix(1,NRF,NU)    # [bs, v]
#     # print("fade rand shape",np.shape(fadeRand))

#     #signal matrix for RF
#     # SRF = gammaI*fadeRand*PR*d_matrix
#     # SRF = np.dot(gammaI,fadeRand)
#     SRF = np.multiply(gammaI,fadeRand)
#     # print("srf 1",np.shape(SRF))

#     # SRF = np.dot(SRF,PR)
#     SRF = np.multiply(SRF,PR)
#     # print("srf 2",np.shape(SRF))

#     # SRF = np.dot(SRF,d_matrix)
#     SRF = np.multiply(SRF,np.transpose(d_matrix))
#     # print("srf 3",np.shape(SRF))
#     '''
#      SRF = np.linalg.matrix_power(SRF,-1*alpha)
#     '''
#     # SRF = fractional_matrix_power(SRF,-1*alpha)
#     SRF = SRF ** (-1 * alpha) #fractional_matrix_power(distance_matrix_thz,2)
#     # print("srf 4",np.shape(SRF))
    
#     # interference : interf=repmat(sum(SRF,1),NRF,1)-SRF; %interference for RF
#     sum_srf = sum_of_each_row(SRF)  # [v]
#     # print("sum_srf",np.shape(sum_srf))

#     # np.tile(sum_srf, (NRF, 1))
#     interf=np.matlib.repmat(sum_srf,NRF,1)
#     # print("interf",np.shape(interf))

#     interf=np.subtract(interf, SRF)
    
#     # print("interf shape",np.shape(interf))
#     # print("interf",interf)

#     #power from all base-stations to all users
#     NP=10e-10 #(10) ** (-10)
#     RPrAllu1 = Wr * np.log2(np.add(1,np.divide(SRF,np.add(NP, interf))))
#     # print(RPrAllu1)
#     RPrAllu1 = np.transpose(RPrAllu1)
#     interf=np.transpose(interf)
#     # print(RPrAllu1.shape)


#     ## column row names should be recovered ### 
#     # print(distance_matrix) 
#     # print(d_matrix)
#     # print(RPrAllu1)
#     # print('vehicle list is ', vehicles,)
#     # print('bs_list is',bss)

#     sinr_matrix = pd.DataFrame(RPrAllu1 , columns = bss, index = vehicles)
#     interf_matrix = pd.DataFrame(interf , columns = bss, index = vehicles)
#     # print(df)

#     return sinr_matrix,interf_matrix


# def rf_sinr_matrix(distance_matrix):
#     """
#     Convert distance matrix to sinr matrix

#     """
#     NU,NRF = distance_matrix.shape # row is vehicle, column is rf bs

#     d_matrix = np.array(distance_matrix)    # [v, bs]

#     fadeRand = generate_exponential_matrix(1,NRF,NU)    # [bs, v]

#     #signal matrix for RF
#     # SRF = gammaI*fadeRand*PR*d_matrix
#     SRF = gammaI * PR * fadeRand * d_matrix.T

#     SRF = SRF ** (-1 * alpha) # [bs, v]

#     interf = SRF.sum(axis=0) - SRF # [bs, v]
    
#     NP=10e-10 #(10) ** (-10)
#     RPrAllu1 = Wr * np.log2(SRF / (NP + interf) + 1).T # [v, bs]
#     interf = interf.T # [v, bs]

#     sinr_matrix = RPrAllu1
#     interf_matrix = interf
#     # print(df)

#     return sinr_matrix,interf_matrix


# def thz_sinr_matrix(distance_matrix):
#     """
#     Convert distance matrix to sinr matrix

#     """
    
#     NU,NTHz = distance_matrix.shape # row is vehicle, column is rf bs

#     d_matrix = np.array(distance_matrix).T

#     fadeRand1 = generate_exponential_matrix(1,NTHz,NU)

#     # SRF = PT * gammaII * fadeRand * d_matrix.T
#     # SRF = SRF ** (-1*alpha) #fractional_matrix_power(distance_matrix_thz,2)
#     # interf = SRF.sum(axis=0) - SRF
#     # NP=10e-10 #(10) ** (-10)
#     # RPrAllu1 = Wr * np.log2(1 + SRF / (NP + interf)).T

#     STHz = gammaII * fadeRand1 * PT * np.exp(-kf * d_matrix) / (d_matrix**2) # signal matrix for THZ
#     interfT = np.tile(np.sum(STHz, axis=0), (NTHz, 1)) - STHz # interference matrix for THz
#     TPrAllu1 = Wt * np.log2(1 + STHz / (NP + interfT))

#     interf = interfT.T

#     sinr_matrix = TPrAllu1
#     interf_matrix = interf

#     return sinr_matrix,interf_matrix

# def sinr_with_threshold(sinr_matrix, bs_assignment):
#     ''' 
#     Input 
#     1.sinr matrix (merged rf thz version)
#     2.bs assignment matrix
#     we devide the sinr matrix to the bss
#                             rf1    rf2    rf3    rf4    rf5   rf6    rf7    th1    th2    th3    th4    th5    th6
#     sinr                    10     20     30     20     20    40     10     20     30     40     40     30     20
#     bss assignment          4      8      9      9      2     10     3      3      3      2      4      1      3
#     data with threshold	    2.5   2.5     10/3   20/9   10    4      10/3   20/3   10     20     10     30     20/3
#     Output
#     sinr matrix with threshold
#     '''
#     sinr_matrix_with_threshold = sinr_matrix / (bs_assignment.sum(axis = 0) + 1e-8)
#     # sinr_matrix_with_threshold = sinr_matrix.div(bs_assignment.sum(),index=sinr_matrix.columns)

#     return sinr_matrix_with_threshold


# '''
# The a2c_link function calculates the signal-to-noise ratio (SNR) for each link between a vehicle and a base station (BS) using the 3D coordinates of the vehicles and BSs, and the distance between them. The function uses the Loss_LoS and theta_ik functions to calculate the path loss and angle of incidence, respectively.

# The Loss_LoS function calculates the path loss using the distance between the vehicle and BS, the carrier frequency, and the environment parameters. The function returns the path loss for both line-of-sight (LoS) and non-line-of-sight (NLoS) scenarios.

# The theta_ik function calculates the angle of incidence between the vehicle and BS using the 3D coordinates of the vehicle and BS. The function returns the angle in radians and degrees.

# The a2c_link function iterates over all vehicle-BS pairs and calculates the path loss and SNR for each link using the Loss_LoS and theta_ik functions. The function returns a matrix of SNR values.

# In summary, the a2c_link function calculates the SNR for each link between a vehicle and a BS using the distance, 3D coordinates, and environment parameters.

# '''

#  ## ===================== Aeriation Parameters ========================================
# eta_LoS = 1
# eta_NLoS = 20
# f_c = 2.1e9 # Carrier Frequency is 2GHZ 850MHz 


# def Loss_LoS(dist_2d,i,k):
#     # link between uav i and BS k
#     dist_ik = dist_2d[i,k]
#     loss_LoS = 20 * np.log( (4 * pi * f_c * dist_ik)/ (3e8) ) + eta_LoS # check natural base or 10 base
#     loss_NLoS = 20 * np.log( (4 * pi * f_c * dist_ik)/ (3e8) ) + eta_NLoS 
#     return loss_LoS, loss_NLoS

# def theta_ik(dist_3d,vehicles_pos_3d,i,k):
#     dist_3d_ik = dist_3d[i,k]
    
#     # if len(vehicles_pos_3d[i, k]) < 3:
#     #     print(f"Vehicle {i} at position {k} has no third element!")
#     #     return None, None
#     # print(dist_3d_ik)
#     h = vehicles_pos_3d[i,2]

#     # calculate the inverse tangent of y/x in radians
#     angle = math.atan(h / dist_3d_ik)
#     # degrees = math.degrees(angle)

#     # print("angle in radians:", angle)
#     # print("angle in degrees:", math.degrees(angle))
#     return angle#, degrees



# def a2c_link(dist_2d,dist_3d,vehicles_pos_3d):

    
#     NU,NRF = dist_2d.shape # row is vehicle, column is rf bs
#     a = -1.5#  a and b are constants that depend on the environment.
#     b = 3.5 #
#     A = eta_LoS - eta_NLoS
#     B = 20 * np.log( (4 * pi * f_c)/ (3e8 ) ) + eta_NLoS 
#     P_Noise = -95.9
#     PT_aeriation = 40

#     P_matrix = np.zeros((NU,NRF))
#     for i, row in enumerate(dist_2d):
#         for k, dist_ik in enumerate(row):
#             r_ik = dist_3d[i,k]
#             # loss_LoS_ik, loss_NLoS_ik= Loss_LoS(dist_2d=dist_2d,i = i,k = k)
#             angleik = theta_ik(dist_3d=dist_3d,vehicles_pos_3d=vehicles_pos_3d,i = i,k = k)
#             loss_ik = A/(1 + a * np.exp(-b *(angleik) - a)) + \
#                     20 * np.log(r_ik/math.cos(angleik)) + B
#             P_matrix[i,k] = PT_aeriation - loss_ik - P_Noise
    