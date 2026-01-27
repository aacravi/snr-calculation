from lisaorbits import EqualArmlengthOrbits
from fastgb import fastgb
import numpy as np

def tdi_XYZ_fastGB(f0:float, f_dot:float, ampl:float, bet: float, lam:float, psi:float, inclination:float, phi0:float, T_obs:float, N:int, delta_t:float, tdi=1.5, **kwargs):
    """
    Generate TDI X,Y,Z channels using FastGB
    
    Parameters
    ==========

    f0: initial frequency in Hz

    f_dot: frequency derivative in Hz^2

    ampl: strain amplitude of the signal (dimensionless)

    bet: ecliptic latitude in radians, [-pi/2, pi/2]

    lam: ecliptic longitude in radians, [0, 2pi]

    psi: polarization angle in radians, [0, pi]

    inclination: inclination angle in radians, [0, pi]

    phi0: initial phase in radians, [0, 2pi]

    T_obs: observation time in seconds

    N: number of frequency bins to generate waveform

    delta_t: sampling time in seconds 
    
    """
    pGB = np.array([f0 , # f0 Hz
                    f_dot, # fdot "Hz^2
                    ampl, # ampl strain
                    bet,  # eclipticlatitude radian b
                    lam,   # eclipticLongitude radian l
                    psi, # polarization radian
                    inclination,  # inclination radian
                    phi0, # initial phase radian
                ])
    if tdi == 1.5:
        # delta_t = sampling time in td
        fgb = fastgb.FastGB(delta_t=delta_t, T=T_obs, N=N, orbits=EqualArmlengthOrbits())
        X, Y, Z, kmin = fgb.get_fd_tdixyz(pGB.reshape(1,-1))
    if tdi == 2.0:
        fgb = fastgb.FastGB(delta_t=delta_t, T=T_obs, N=N, orbits=EqualArmlengthOrbits())
        X, Y, Z, kmin = fgb.get_fd_tdixyz(pGB.reshape(1,-1), tdi2=True)
    df = 1/T_obs
    fr =(np.arange(N) + kmin[0])*df

    return X, Y, Z, kmin, fr

def tdi_AE_fastGB(f0:float, f_dot:float, ampl:float, bet: float, lam:float, psi:float, inclination:float, phi0:float, T_obs:float, N:int, delta_t:float, tdi =1.5,**kwargs):
    """
    Generate TDI A, E channels using FastGB, from the X, Y and Z channels
    
    Parameters
    ==========

    f0: initial frequency in Hz

    f_dot: frequency derivative in Hz^2

    ampl: strain amplitude of the signal (dimensionless)

    bet: ecliptic latitude in radians, [-pi/2, pi/2]

    lam: ecliptic longitude in radians, [0, 2pi]

    psi: polarization angle in radians, [0, pi]

    inclination: inclination angle in radians, [0, pi]

    phi0: initial phase in radians, [0, 2pi]

    T_obs: observation time in seconds

    N: number of frequency bins to generate waveform

    delta_t: sampling time in seconds 
    
    """
    X, Y, Z, kmin, fr = tdi_XYZ_fastGB(f0, f_dot, ampl, bet, lam, psi, inclination, phi0, T_obs, N, delta_t, tdi)
    A =  1/np.sqrt(2) * (Z - X)
    E = 1/np.sqrt(6)* (X -2*Y + Z)
    return A, E, kmin, fr


def tdi_XYZ_fastGB_multi(params_array, delta_t, T_obs, N, tdi=1.5):
    if tdi == 1.5:
        fgb = fastgb.FastGB(delta_t=delta_t, T= T_obs, N=N, orbits=EqualArmlengthOrbits())
        X, Y, Z, kmin = fgb.get_fd_tdixyz(params_array)
    if tdi == 2.0:
        fgb = fastgb.FastGB(delta_t=delta_t, T= T_obs, N=N, orbits=EqualArmlengthOrbits())
        X, Y, Z, kmin = fgb.get_fd_tdixyz(params_array, tdi2=True)
    df = 1/T_obs
    base = np.arange(N)  

    fr = (base[np.newaxis, :] + kmin[:, np.newaxis]) * df
    return X, Y, Z , kmin, fr

def tdi_AE_fastGB_multi(params_array, delta_t, T_obs, N, tdi =1.5):
    X, Y, Z, kmin, fr = tdi_XYZ_fastGB_multi(params_array, delta_t, T_obs, N, tdi)
    A =  1/np.sqrt(2) * (Z - X)
    E = 1/np.sqrt(6)* (X -2*Y + Z)
    return A, E, kmin, fr
