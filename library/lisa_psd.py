# --- noise psd model for the data analysis ----
import scipy.constants as constants
import numpy as np

def C_xx(omega,  tdi, L= 2.5e9/constants.c, **kwargs):
    if tdi == 1.5:
        Cxx = 4 * np.sin(omega * L)**2
    if tdi == 2.0:
        Cxx = 16 * (np.sin(omega * L) ** 2) * (np.sin(2 * omega * L) ** 2)
    return Cxx
 
def C_xy(omega, L= 2.5e9/constants.c):
    return -16 * (np.sin(omega * L)) * (np.sin(2 * omega * L) ** 3)
 
# Transfer functions fot the AE channels
 
def TransferAE_acc(omega, tdi, L= 2.5e9/constants.c, **kwargs):
    return 4* C_xx(omega, tdi, L) * (3 + 2 * np.cos(omega * L) + np.cos(2 * omega * L))
 
def TransferAE_OMS(omega, tdi, L= 2.5e9/constants.c, **kwargs):
    return 2 * C_xx(omega, tdi, L) * (2 + np.cos(omega * L))
 
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
 
 
def S_acc_AE(f, tdi, L= 2.5e9/constants.c, **kwargs):
    return TransferAE_acc(2 * np.pi * f, tdi, L) * S_acc(f)
 
 
def S_OMS_AE(f, tdi, L= 2.5e9/constants.c, **kwargs):
    return TransferAE_OMS(2 * np.pi * f, tdi, L) * S_OMS(f)

def  S_gal(f, L= 2.5e9/constants.c,):
    omega = 2 * np.pi * f
    A = 9e-45
    fk = 0.00113 
    alpha = 0.138
    beta = -221
    k = 521
    gam = 1680

    return A*f**(-7/3)*np.exp(-f**alpha + beta*f*np.sin(k*f))*(1+np.tanh(gam*(fk-f)))

def  S_gal2(f, T_obs, L= 2.5e9/constants.c, **kwargs):
    # T_obs needs to be in yr, given in seconds for FastGB
    if T_obs > 10:
        T_obs = T_obs/ (365*24*3600)
    omega = 2*np.pi*f
    A = 1.15e-44
    log_f_knee = -0.37*np.log10(T_obs)-2.49
    log_f1 = -0.15*np.log10(T_obs) -2.72
    alpha = 1.56
    f1 = 10**log_f1
    f_knee = 10**log_f_knee
    s2 = 0.00067
    return A * (f)**(-7/3) * np.exp(-(f/f1)**alpha) * (1+np.tanh(-(f-f_knee)/s2))

def S_gal_response(f, tdi, L= 2.5e9/constants.c, **kwargs):
    """
    Response to galactic noise for channels X,Y,Z and TDI 1.5 or 2.0
    """

    omega = 2*np.pi*f
    if tdi ==1.5:
        response = 2*(omega*L)**2 * np.sin(omega*L)**2 
    if tdi == 2.0:
        response = 2*(omega*L)**2 * np.sin(omega*L)**2 * 4*np.sin(2*omega*L)**2
    return response


def noise_psd_AE( f, tdi, L= 2.5e9/constants.c, **kwargs):
    """
    LISA psd for A,E TDI channels without galactic confusion

    =======
    f: frequency array

    L: armlength in seconds
    """
    if str(tdi) not in ["1.5", "2.0"]:
        raise ValueError("The version of TDI, currently only for 1.5 or 2.0.")
    
    return S_acc_AE(f, tdi, L) + S_OMS_AE(f, tdi, L)

def noise_psd_XYZ(f, L= 2.5e9/constants.c, **kwargs):
    """
    LISA psd for X,Y,Z TDI channels

    ======
    f: frequency array

    L: armlength in seconds
    """
    omega = 2 * np.pi * f
    return 2* (np.sin(omega*L))**2 *(S_OMS(f) + (3+ np.cos(2*omega*L))*S_acc(f))


def noise_psd_AE_gal(f, tdi, L= 2.5e9/constants.c, **kwargs):
    return noise_psd_AE(f, tdi, L) + 1.5*S_gal_response(f, tdi, L)*S_gal(f,L)


def noise_psd_AE_gal2(f, T_obs, tdi, L= 2.5e9/constants.c,  **kwargs):
    """
    LISA PSD for A,E TDI channel with galactic confusion noise

    =====
    f: frequency array

    L: armelngth in seconds

    T_obs: observation time in yr

    tdi: the TDI generation (1.5 or 2.0)

    ====
    Note: for galactic foreground, PSD_A/E = 1.5 PSD_X
    """
    if str(tdi) not in ["1.5", "2.0"]:
        raise ValueError("The version of TDI, currently only for 1.5 or 2.0.")
    
    return noise_psd_AE(f, tdi, L) + 1.5*S_gal_response(f, tdi, L)* S_gal2(f, T_obs, L)

from library.sgwb_Boileau import sgwb_noise_boileau
from pycbc.psd.analytical_space import averaged_response_lisa_tdi
def noise_psd_AE_Boileau(f,  tdi, model, **kwargs):
    return  1.5*averaged_response_lisa_tdi(f, tdi =tdi)* sgwb_noise_boileau(f, model)


def S_noise_approx(f):
    f1 = 0.0004
    f2 = 0.025
    S_I = 5.76e-48 * (1 + (f1/f)**2)
    S_II = 3.6e-41
    R = (1 + (f/f2)**2)
    S_noise = 10/3 * (S_I/(2 * np.pi * f)**4 + S_II) * R
    return S_noise

def S_from_h_char(h_c, f0):
    psd = h_c**2 / f0
    return psd

def psd_source_approx(h_c, f0, tdi, L= 2.5e9/constants.c, **kwargs):
    psd = S_from_h_char(h_c, f0) *  1.5 * S_gal_response(f0, tdi, L)
    return psd
