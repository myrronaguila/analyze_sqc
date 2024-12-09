# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:58:58 2024

Effective fitting model for deterministic benchmarking - lmfit.minimize

@author: Mai
"""

import numpy as np
from uncertainties import ufloat
from uncertainties.core import Variable  # This is the base type for ufloat

"""lineshapes supporting det benchmarking"""
"""----------------------------Basic Functions------------------------------"""
def cosine(x, amplitude=1.0, frequency=1.0, shift=0):
    """Return a cosine function. #not noted - same notation as LMFIT

    cosine(x, amplitude, frequency, shift) =
        amplitude * sin(x*frequency + shift)

    """
    return amplitude*np.cos(x*frequency+shift)

def decay_cos_c(t, T0, fd, Ioffs, Iamp):
    """
    general fitting model for decaying cosine function with y-offset.
    
    (1-a) -> amp 
    
    Parameters
    ----------
    t : 1D numpy array
        time depending on purpose.
    T0 : float
        characteristic time.
    fd : float
        drive frequency.
    Ioffs : float
        Offset value.
    Iamp : float
        Amplitude of oscillation.
    phi_I : float
        Phase offset in Radians.

    Returns
    -------
    1D numpy array.
    """
    wd = 2*np.pi*fd
    return Ioffs + Iamp*np.cos(wd*t)*np.exp(-(t/T0))

"""---------------------------------Main Functions-------------------------"""
def dB_errors(tn, f_e, T_d, a):
    """
    deterministic benchmarking determination of errors

    Parameters
    ----------
    tn : float / 1D numpy array
        Evolution time (in ns).
    f : float / frequency
        frequency of error.
    T_d : float
        characteristic lifetime (in ns).
    a : float
        temperature factor.

    Returns
    -------
    F : TYPE
        DESCRIPTION.
    """
    
    omega = 2*np.pi*f_e
    F = 0.5*(1+a)+0.5*(1-a)*np.exp(-tn/T_d)*np.cos(2*omega*tn)
    return F

def dB_fidelity(n, A, B, r_clif, num_qubits):
    """
    RB-like fidelity calculations for deterministic benchmarking (after test 1 to 4)
    

    Parameters
    ----------
    n : 1D numpy array
        number of cliffords
    A : float
        initial probability.
    B : float
        saturation probability.
    r_clif : float
        Clifford Error.
    num_qubits : int
        number of qubits

    Returns
    -------
    F : 1D numpy array
        Fidelity equation.

    """
    num_qubits = int(num_qubits)
    d = 2**num_qubits
    #r_clif = (d-1)*(1-p)/d
    p = 1- d*r_clif/(d-1)
    F = A*np.power(p, n) + B
    return F

"""Data extraction"""
# def check_dtype(value):
#     if isinstance(value, Variable):  # Check if it's a ufloat
#         return "The input value is of type ufloat."
#     elif isinstance(value, float):  # Check if it's a standard float
#         return "The input value is of type float."
#     else:
#         return "The input value is neither ufloat nor float."

def get_db_errors(f, tg, err_type='rot'):
    """
    identify benchmarking errors

    Parameters
    ----------
    f : float / unc
        Frequency in Hz.
    tg : float
        gate time in seconds.
    err_type : string, optional
        Rotation or phase errors. The default is 'rot'.

    Returns
    -------
    Alignment errors in degrees (very small value in radians).
    """
    omega = 2*np.pi*f
    # check dtype of omega and check if error is rotational of phase
    if err_type == 'rot':
        # rotation error from YY
        # deg_err = np.rad2deg(omega*2*tg) # page 1
        # deg_err = np.rad2deg(omega*2*tg) # problem with ufloat
        deg_err = (omega*2*tg)*180/np.pi
    else:
        # phase error from XXbar
        # deg_err = np.rad2deg(omega*tg) # page 1
        # deg_err = np.rad2deg(omega*tg) # problem with ufloat
        deg_err = (omega*tg)*180/np.pi
    return deg_err

def get_1Q_gate_error(p, num_qubits=2):
    """
    RB-like expression for fidelity

    Parameters
    ----------
    p : float / unc
        gate fidelity.
    d : float / integer, optional
        number of hilbertspace. The default is 2.

    Returns
    -------
    return gate error with uncertainties.

    """
    d = np.power(2, num_qubits)
    r = (1-p)*(d-1)/d
    return r

def get_spam_error(A, B):
    """
    Get state preparation and measurement error

    Parameters
    ----------
    A : float
        initial state probablity.
    B : float
        final state probability.

    Returns
    -------
    SPAM error.
    """
    return 1-(A+B)
    