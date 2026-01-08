# file to include Violeau et al. SGWB from http://arxiv.org/abs/2506.18390

import numpy as np

def sgwb_Omega_Boileau_model(f, A, fb, a1, D, a2, **kwargs):
    return A*(f/fb)**(-a1) * (1/2*(1+(f/fb)**(1/D)))**((a1-a2)*D)

models = {
    "Madau Dickinson alpha4":{
        "A": 10e-12,
        "fb": 0.007,
        "a1": -0.69,
        "a2": 2.74,
        "D": 0.26
    },
    "Madau Fragos default":{
        "A": 4e-12,
        "fb": 0.007,
        "a1": -0.72,
        "a2": 2.41,
        "D": 0.24
    },
    "Strolger alpha4":{
        "A": 30e-12,
        "fb": 0.010,
        "a1": -0.69,
        "a2": 5.11,
        "D": 0.31
    }
}

def sgwb_omega_boileau(f, model):
    """
    Returns fit for Omega for the chosen model

    Possible models: Madau Dickinson alpha4, Madau Fragos default, Strolger alpha4
    """
    params = models[model]
    return sgwb_Omega_Boileau_model(f, params["A"], params["fb"], params["a1"], params["D"], params["a2"])

def sgwb_noise_boileau(f, model):
    omega = sgwb_omega_boileau(f,model)
    H = 67.66 / 3.086e19
    return omega * 3 * H**2/(4 * np.pi**2 * f**3)