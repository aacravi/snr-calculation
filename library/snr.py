import numpy as np

def noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    =======
    Noise-weighted inner product.
    """

    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 / duration * integrand.sum()

def optimal_snr_squared(signal, power_spectral_density, duration):
    """
    Compute the square of the optimal matched filter SNR for the provided
    signal.


    Parameters
    ==========
    signal: array_like
        Array containing the signal
    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The optimal matched filter signal to noise ratio squared

    """
    return noise_weighted_inner_product(signal, signal, power_spectral_density, duration)


def matched_filter_snr(signal, frequency_domain_strain, power_spectral_density, duration):
    """
    Calculate the _complex_ matched filter snr of a signal.
    This is <signal|frequency_domain_strain> / optimal_snr

    Parameters
    ==========
    signal: array_like
        Array containing the signal
    frequency_domain_strain: array_like

    power_spectral_density: array_like

    duration: float
        Time duration of the signal

    Returns
    =======
    float: The matched filter signal to noise ratio squared

    """
    rho_mf = noise_weighted_inner_product(
        aa=signal, bb=frequency_domain_strain,
        power_spectral_density=power_spectral_density, duration=duration)
    rho_mf /= optimal_snr_squared(
        signal=signal, power_spectral_density=power_spectral_density,
        duration=duration)**0.5
    return rho_mf

def optimal_snr(signal, power_spectral_density, T_obs, **kwargs):
    """
    Compute the optimal matched filter SNR for the provided signal.

    Parameters
    ==========
    signal: array_like
        Array containing the signal
    power_spectral_density: array_like

    T_obs: float
        Time duration of the signal

    Returns
    =======
    float: The optimal matched filter signal to noise ratio

    """
    return (optimal_snr_squared(
        signal=signal, power_spectral_density=power_spectral_density,
        duration=T_obs).real)**0.5

from library.lisa_psd import S_noise_approx
def approx_snr(h_c , f):
    S_noise = S_noise_approx(f)
    SNR = h_c /(f* S_noise)**(1/2)
    return SNR


def optimal_snr_AE(A, E, psd,  T_obs, **kwargs):
    snr = (optimal_snr_squared(A, psd, T_obs).real + optimal_snr_squared(E, psd, T_obs).real)**(1/2)
    return snr
