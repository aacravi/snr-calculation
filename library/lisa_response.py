import os
os.environ["NO_PKGCONFIG"] = "1"
import pycbc
import numpy as np
from pycbc.types import FrequencySeries
from pycbc.waveform import get_fd_waveform


# clean data: replace Nans with zeros
def clean_fdata(h):
    """
    Clean FrequencySeries data by replacing NaNs with zeros.
    """

    data = h.data
    real = np.nan_to_num(data.real, nan=0.0)
    imag = np.nan_to_num(data.imag, nan=0.0)
    clean_data = real + 1j*imag 

    return FrequencySeries(clean_data, delta_f=h.delta_f)


def waveform_fd(apx: str, mass1: float, mass2: float, fMin: float, fMax: float, delta_f: float, d_luminosity: float, inclination: float, **kwargs):
    """
    Generate waveform in frequency domain using PyCBC
    
    Parameters
    ==========
    apx: waveform approximant

    mass1: in solar masses

    mass2: in solar masses

    fMin: minimum frequency in Hz of waveform

    fMax: maximum frequency in Hz of waveform

    delta_f: frequency step in Hz

    d_luminosity: luminosity distance in Mpc, default = 1Mpc

    inclination: inclination angle in radians
    """
    hp_, hc_ = get_fd_waveform(
        approximant=apx,
        mass1=mass1,
        mass2=mass2,
        f_lower=fMin,
        delta_f=delta_f,
        f_final = fMax,
        distance = d_luminosity, # in Mpc
        inclination = inclination
    )

    hp = clean_fdata(hp_)
    hc = clean_fdata(hc_)

    return hp, hc


# -----------------------------------------------------------------------------------------
#RESPONSE FUNCTIONS

# Set constants: Speed of light, year in seconds, AU in seconds, LISA armlength in seconds
c = 299792458.0
year = 31558149.763545603
AUs = 499.00478383615643
L = 2.5e9/c

tm = np.arange(0, year, 1000)

# LISA motion
def LISAmotion(tm):
    """
    Calculate the positions and link unit vectors of the LISA spacecrafts at time tm (s)
    """
    lamb = 0.0
    kappa = 0.0

    N = len(tm)

    a = AUs
    e = L/(2.0*np.sqrt(3.0)*a) #eccentricity
    nn = np.array([1,2,3])

    Beta = (nn-1)*2.0*np.pi/3.0 + lamb # position of the spacecrafts
    alpha = 2.0*np.pi*tm/year + kappa  # LISA CoM position

    # x,y,z coordinates of 3 satellites in SSB coordinates
    x = np.zeros((3, N))
    y = np.zeros((3, N))
    z = np.zeros((3, N))
    for i in range(3):
        x[i, :] = a*np.cos(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.sin(Beta[i]) - (1.0 + (np.sin(alpha))**2)*np.cos(Beta[i]))
        y[i, :] = a*np.sin(alpha) + a*e*(np.sin(alpha)*np.cos(alpha)*np.cos(Beta[i]) - (1.0 + (np.cos(alpha))**2)*np.sin(Beta[i]))
        z[i, :] = -np.sqrt(3.0)*a*e*np.cos(alpha - Beta[i])
        
    ### Links
    n23 = np.array([x[1,:]-x[2,:], y[1,:]-y[2,:], z[1,:] - z[2,:]])/L
    n31 = np.array([x[2,:]-x[0,:], y[2,:]-y[0,:], z[2,:] - z[0,:]])/L
    n12 = np.array([x[0,:]-x[1,:], y[0,:]-y[1,:], z[0,:] - z[1,:]])/L

    ## Vector position of satellites
    r1 = np.array([x[0,:], y[0,:], z[0,:]])
    r2 = np.array([x[1,:], y[1,:], z[1,:]])
    r3 = np.array([x[2,:], y[2,:], z[2,:]])

    return ((n23, n31, n12), (r1, r2, r3))


t = 100 # time to calculate LISA position in s
n23, n31, n12 = (LISAmotion(np.array([t]))[0]) # unit vectors at time t 
#print(n23)
r1, r2, r3 = (LISAmotion(np.array([t]))[1]) # position vectors at time t


# Response functions 
def Fp_rs(psi: float, lam: float, bet: float, n, **kwargs):
    """
    Calculate the plus polarization response function for a given source location and polarization angle.

    Parameters
    ==========

    psi: polarization angle, [0, pi]

    lam: ecliptic longitude, [0, 2pi]

    bet: ecliptic latitude, [-pi/2, pi/2]

    n: unit vector receiver-sender, n23, n31, n12 
    """
    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(bet)*np.cos(lam), -np.sin(bet)*np.sin(lam), np.cos(bet)])
    
    nu = np.dot(u, n)
    nv = np.dot(v, n)
        
    plus = nu*nu - nv*nv
    cros = 2.*nu*nv
    
    Fp = np.cos(psi)*plus + np.sin(psi)*cros
    
    return (Fp)
   

def Fc_rs(psi: float, lam: float, bet: float, n: np.ndarray, **kwargs):
    """
    Calculate the cross polarization response function for a given source location and polarization angle.

    Parameters
    ==========

    psi: polarization angle, [0, pi]

    lam: ecliptic longitude, [0, 2pi]

    bet: ecliptic latitude, [-pi/2, pi/2]

    n: unit vector receiver-sender, n23, n31, n12
    """

    u = np.array([np.sin(lam), -np.cos(lam), 0.0])
    v = np.array([-np.sin(bet)*np.cos(lam), -np.sin(bet)*np.sin(lam), np.cos(bet)])
    
    nu = np.dot(u, n)
    nv = np.dot(v, n)
    
    plus = nu*nu - nv*nv
    cros = 2.*nu*nv
    
    Fc = -np.sin(psi)*plus + np.cos(psi)*cros
    
    return (Fc)

def Psi(lam: float, bet: float, n: np.ndarray, f, **kwargs):
    
    # f = array of frequencies of GW
    ### direction of GW propagation
    k = -1.*np.array([np.cos(bet)*np.cos(lam), np.cos(bet)*np.sin(lam), np.sin(bet)])
    
    kn = np.dot(k, n)

    x = 2*np.pi*f*L
    
    Gam_rs = np.sinc( (1.-kn)*0.5*x/np.pi ) *np.exp(-0.5j*x*(1.-kn))
    Gam_sr = np.sinc( (1.+kn)*0.5*x/np.pi ) *np.exp(-0.5j*x*(1.+kn))
    
    Psi_rs = Gam_rs + Gam_sr*np.exp(-1.j*x*(1-kn))
    
    return (Psi_rs) # returns an array


def tdiX_15(hplus: FrequencySeries, hcross: FrequencySeries, psi: float, lam: float, bet: float, t: float, **kwargs):
    """
    Calculate the 1.5 generation TDI X channel response to given plus and cross polarization waveforms.
    
    Parameters
    ==========
    hplus: + polarization FrequencySeries

    hcross: x polarization FrequencySeries

    psi: polarization angle, [0, pi]

    lam: ecliptic longitude, [0, 2pi]

    bet: ecliptic latitude, [-pi/2, pi/2]

    t: time to calculate LISA position in s
    """
    # hplus,hcross: FrequencySeries
    f = hplus.sample_frequencies
    om = 2*np.pi*f
    x = om*L

    # time to calculate LISA position in s
    n23, n31, n12 = (LISAmotion(np.array([t]))[0]) # unit vectors at time t 
    #print(n23)
    r1, r2, r3 = (LISAmotion(np.array([t]))[1]) # position vectors at time t

    k = -1.*np.array([np.cos(bet)*np.cos(lam), np.cos(bet)*np.sin(lam), np.sin(bet)])
    kr = np.dot(k, r1)

    n13 = -n31

    Fp = Fp_rs(psi=psi, lam=lam, bet=bet, n=n13)*Psi(lam=lam, bet=bet, n=n13, f=f) - Fp_rs(psi=psi, lam=lam, bet=bet, n=n12)*Psi(lam=lam, bet=bet, n=n12, f=f)
    Fc = Fc_rs(psi=psi, lam=lam, bet=bet, n=n13)*Psi(lam=lam, bet=bet, n=n13, f=f) - Fc_rs(psi=psi, lam=lam, bet=bet, n=n12)*Psi(lam=lam, bet=bet, n=n12, f=f)
    X = x*np.sin(x)* np.exp(-1j*(om*(t - kr)-x))*(hplus*Fp + hcross*Fc)

    return X
# f = hp.sample_frequencies
