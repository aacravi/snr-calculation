# --- noise psd model for the data analysis ----
import scipy.constants as constants
import numpy as np

def C_xx(omega, L, tdi, **kwargs):
    if tdi == 1.5:
        Cxx = 4 * np.sin(omega * L)**2
    if tdi == 2.0:
        Cxx = 16 * (np.sin(omega * L) ** 2) * (np.sin(2 * omega * L) ** 2)
    return Cxx
 
def C_xy(omega, L):
    return -16 * (np.sin(omega * L)) * (np.sin(2 * omega * L) ** 3)
 
# Transfer functions fot the AE channels
 
def TransferAE_acc(omega, L, tdi, **kwargs):
    return 4* C_xx(omega, L, tdi) * (3 + 2 * np.cos(omega * L) + np.cos(2 * omega * L))
 
def TransferAE_OMS(omega, L, tdi, **kwargs):
    return 2 * C_xx(omega, L, tdi) * (2 + np.cos(omega * L))
 
def S_acc(f):
    A_acc = 3 * 1e-15
    f_acc = 4 * 1e-4
    S_acc = (
        A_acc**2
        * (1 / (2 * np.pi * f * constants.c)**2)
        * (1 + (f_acc / f) ** 2)
        * (1 + (f / (8 * 1e-3)) ** 4)
    )
    return S_acc
 
def S_OMS(f):
    A_oms = 15 * 1e-12
    f_oms = 2 * 1e-3
    S_oms = A_oms**2 * ((2 * np.pi * f) / (constants.c)) ** 2 * (1 + (f_oms / f) ** 4)
    return S_oms
 
 
def S_acc_AE(f, L, tdi, **kwargs):
    return TransferAE_acc(2 * np.pi * f, L,tdi) * S_acc(f)
 
 
def S_OMS_AE(f, L, tdi, **kwargs):
    return TransferAE_OMS(2 * np.pi * f, L, tdi) * S_OMS(f)

def  S_gal(f,L):
    omega = 2 * np.pi * f
    A = 9e-45
    fk = 0.00113 
    alpha = 1.138
    beta = -221
    k = 521
    gam = 1680

    return 2*(omega*L)**2 * np.sin(omega*L)**2*A*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(k*f))*(1+np.tanh(gam*(fk-f)))

def  S_gal2(f, T_obs, L, **kwargs):
    omega = 2*np.pi*f
    A = 1.15e-44
    log_f_knee = -0.34*np.log10(T_obs)-2.53
    log_f1 = -0.22*np.log10(T_obs) -2.78
    alpha = 1.66
    f1 = 10**log_f1
    f_knee = 10**log_f_knee
    s2 = 4.8108e-4

    return 2*(omega*L)**2 * np.sin(omega*L)**2 *A * (f)**(-7/3) * np.exp(-(f/f1)**alpha) * (1+np.tanh(-(f-f_knee)/s2))


def noise_psd_AE( f, L, tdi, **kwargs):
    """
    LISA psd for A,E TDi channels

    =======
    f: frequency array

    L: armlength in seconds
    """
    return S_acc_AE(f, L, tdi) + S_OMS_AE(f, L, tdi)

def noise_psd_XYZ(f, L, **kwargs):
    """
    LISA psd for X,Y,Z TDI channels

    ======
    f: frequency array

    L: armlength in seconds
    """
    omega = 2 * np.pi * f
    return 2* (np.sin(omega*L))**2 *(S_OMS(f) + (3+ np.cos(2*omega*L))*S_acc(f))


def noise_psd_AE_gal(f, L,tdi, **kwargs):
    return noise_psd_AE(f, L, tdi) + S_gal(f,L)


def noise_psd_AE_gal2(f, L, T_obs,tdi,  **kwargs):
    return noise_psd_AE(f, L, tdi) + S_gal2(f, T_obs, L)

