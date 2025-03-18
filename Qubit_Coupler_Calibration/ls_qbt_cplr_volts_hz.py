# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:45:01 2025

Fit function for voltage to qubit frequency / flux portion for qubit
- derived from qbt_cplt_volts_hz.py which is a developing platform
    change-log (20240318)
    - python script with sorted out analytic models
    - sorted out obsolute models for enhanced readability
    - focused on bare minimum reproducibility of key models
    - removed # documentation for better readability and dissemination
- future works:
    - refined name scheme for readability and non-confusion.
    - separate python file that includes a qutip / numerical diagonalization analysis
    - may include ZZ-interaction / iSWAP analysis for the bridge structure.
    
@author: Mai
"""

import numpy as np

from scipy.special import mathieu_a, mathieu_b
from scipy.optimize import minimize

# fundamental constants
h = 6.62607015E-34 # J*s
hbar = 2*np.pi*h 
e = 1.602176634E-19 #C
phi0 = h/(2*e)

"""----------------------------Transmon Equation----------------------------"""
def transmon_eigen(f_01, f_Ec, m=1):
    """
    Eigen-energies in the transmon limit at higher levels

    Parameters
    ----------
    f_01 : float
        maximum sweet-spot frequencies of the 01 transition (Hz, GHz).
    f_Ec : float
        charging energy (negative of anharmonicity, Hz, GHz).
    m : float
        Energy level. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    res = (f_01+f_Ec)*(m+0.5)-(f_Ec/12)*(6*np.power(m,2) + 6*m + 3)
    res -= 0.5*(f_01+f_Ec) - (f_Ec/4)
    return res

def fj_from_f01_transmon(f01, fc):
    """
    Calculate Josepshon Energy (in Hz unit) from f01 and fc => Charging energy from spectroscopy

    Parameters
    ----------
    f01 : float
        01 transition frequency (in Hz).
    fc : float
        charging energy (anharmonicity.

    Returns
    -------
    float
        Frequency in the transmon regime.
    """
    return np.power(f01-fc,2)/(8*fc)

def transmon_eigenvalues(fj, fc, m=10, corr='Y'):
    """
    Compute the eigenvalues of a transmon qubit using Mathieu functions.

    Parameters:
        fc (float): Charging energy of the transmon (in Hz).
        fj (float): Josephson energy of the transmon (in Hz).
        m (int): Number of energy levels to compute.

    Returns:
        eigenvalues (list): List of eigenvalues in Hz.
    """
    # Compute the dimensionless ratio: Ej / Ec
    q = -fj / (2*fc)
    
    eigenvalues = [mathieu_a(i,q)*fc if i%2==0 else mathieu_b(i,q)*fc for i in range(m)]
    eigenvalues = np.array(eigenvalues)
    # correction
    if corr=='Y':
        eigenvalues -= eigenvalues[0]
    
    return eigenvalues

def fj_from_f01_m(f01, fc, m=10):
    """
    Applicable for charge-sensitive box hamiltonian (with matthieu function)
    # determine fj from frequency of the qubit
    
    Use cases:
        1) fit voltage <=> qubit-coupler frequency 
        2) looking for iswap and CZ points

    Parameters
    ----------
    f01 : float
        01 qubit frequency in Hz.
    fc : float
        Charging frequency in Hz.
    m : integer, optional
        Number of energy levels. The default is 10.
    
    Returns
    -------
    Estimate fj.
    """
    
    def cost_func(fj, f01, fc, m):
        # cost function of fj hamiltonian
        return np.abs(f01-transmon_eigenvalues(fj=fj, fc=fc, m=m)[1])
    
    # use 1D optimization for fj
    fj_init = fj_from_f01_transmon(f01, fc)
    # print(fj_init) # check for validity, 24,031,982,545.075253 Hz => 24 GHz
    
    # Call scipy.optimize.minimize with bounds
    result = minimize(cost_func, x0=fj_init,
        args=(f01, fc, m),
        bounds=((1*fc, 10000*fc),),  # Bounds for fj
        method='Nelder-Mead', tol=1e-6    # Optimization method suitable for bounded problems
    )
    
    """
    20250306 -> had this error: for fit with lower values
    OptimizeWarning: Initial guess is not within the specified bounds
      result = minimize(cost_func, x0=fj_init,
    """
    
    return result.x[0] #get one result 

def d_to_fact(d):
    """
    Convert Literature d to fact=Ej2/Ej1
    """
    return (1+d)/(1-d)

def Ej1_Ej2_d(Ejmax, gam):
    """
    Extract Ej1, Ej2 and d from the split transmon
    Note: fact = gam
    input:
        Ejmax : float / ufloat
        josephson energy
        gam : float / ufloat
        ratio of EJ2/Ej1
    """
    Ej1 = Ejmax/(gam+1)
    Ej2 = Ejmax - Ej1
    d = (gam-1)/(gam+1)
    return Ej1, Ej2, d

def M_flux_Line(A_conv, Z0=50):
    """Based on Josephson Inductance"""
    return phi0*A_conv*Z0

"""20250219 - Recasted Analysis---------------------------------------------"""
def f01_volts_to_hz(vpk, A_conv, v0=0, f_max=5E9, fact=1, f_Ec=200E6):
    """
    short-hand method on determining best fit for voltage-to-frequency conversion
    
    np.sqrt(f_Ejmax*f_ec)=f_max+f_ec
    
    Parameters
    ----------
    vpk : float / 1D numpy array
        flux line bias (regardless of pulsed or not) in volts.
    A_conv : 
        Period dictating the flux periodicity (in rad/volts) (np.pi/Vc).
    v0 : float
        offset flux voltage (in volts).
    f_max : float
        qubit frequency in Hz.
    f_Ec : float
        qubit Ec in Hz.
    fact : float
        ratio of Ej2/Ej1 (size factor) (reduces range from 0 to 100)

    Returns
    -------
    f_01_volts: 1D numpy array.
        frequency as a function of DC bias in the transmon regime in Hz
    """
    # initial parameters
    d = (fact-1)/(fact+1)
    
    # model
    A = (f_max + f_Ec) #np.sqrt(8Ejmax*Ec)
    B = np.sqrt(np.abs(np.cos(A_conv*(vpk-v0))))
    C = np.power(1+np.power(d*np.tan(A_conv*(vpk-v0)),2),0.25)
    
    f01_hz=A*B*C-f_Ec
    return f01_hz

def f01_hz_to_volts(f01, A_conv, v0=0, f_max=5E9, fact=1, f_Ec=200E6):
    """
    short-hand method on determining best corresponding voltage for appropriate bias frequency
    
    np.sqrt(f_Ejmax*f_ec)=f_max+f_ec
    
    note that output will be a list of negative and positive voltages. 
        
    Use case:
        idling voltage for qubit / coupler after getting a fit parameter
    
    Parameters
    ----------
    f01 : float / 1D numpy array
        peak frequency.
    A_conv : 
        Period dictating the flux periodicity (in rad/volts) (np.pi/Vc).
    v0 : float
        offset flux voltage (in volts).
    f_max : float
        qubit frequency in Hz.
    f_Ec : float
        qubit Ec in Hz.
    fact : float
        ratio of Ej2/Ej1 (size factor) (reduces range from 0 to 100)

    Returns
    -------
    zpa_volts: [-float,+float]
        flux bias in -negative and positive voltage
    """
    # initial parameters
    d = (fact-1)/(fact+1)
    A = (f_max + f_Ec) #np.sqrt(8Ejmax*Ec)
    
    # model
    A = np.power((f01+f_Ec)/(f_max+f_Ec),4)
    B = 1+d**2
    C = 1-d**2
    D = 1/(2*A_conv)
    volts = D*np.arccos((2*A-B)/C)
    
    f01_volt_n = v0-volts
    f01_volt_p = v0+volts
    return [f01_volt_n, f01_volt_p]

"""--------------------------Anticrossing Physics---------------------------"""
def anticrossing_model(f_tune, f_fixed, g_hz):
    """
    Model for anti-crossing physics - useful in scanning SWAP between coupler and Q1
    20250306 - required concatenation, easy to model, unfortunately, hard to 
    do 1-1 confirmation between f_tune for upper and lower branches, 
    
    Use cases: 
        1. Fast simulation of anti-crossing model
        2. Visualization of parameters according f_tune.
    Non-use case:
        1. Fit model (as both lmfit.Model and lmfit.minimize requires 
                      1-to-1 correspondence between xdata and ydata)
    
    Parameters
    ----------
    f_tune : 1D numpy array
        Bare mode tuned frequency.
    fq_fixed : float
        Fixed (or idle) qubit / resonator / coupler frequency in Hz.
    g_hz : float
        Coupling strength in Hz.

    Returns
    -------
    np.concatenate(upper and lower branch)
        1D numpy array consisting of upper and lower branch.
        length is twice that of f_tune.

    """
    A = (f_fixed+f_tune)/2
    B = np.sqrt(np.power(g_hz,2)+ np.power((f_fixed-f_tune)/2,2))
    f_plus, f_minus = A + B, A - B
    return np.concatenate((f_plus, f_minus)) # upper branch, then lower branch.

"""--Anticrossing mod2 and res2- good in visualizing sequential data but bad 
in considering multiple two branch data in multiple anti-crossings----------"""

def anticrossing_mod2(f_tune, f_fixed, g_hz, y_ref):
    # ver2 -> add y-data for reference, working for unique f_tune and branches
    A = (f_fixed+f_tune)/2
    B = np.sqrt(np.power(g_hz,2)+ np.power((f_fixed-f_tune)/2,2))
    f_plus, f_minus = A + B, A - B
    # Find the closest branch to each y_data point
    return np.where(y_ref > f_fixed, f_plus, f_minus)  # Match structure of y_data

def anticrossing_res2(params, f_tune, data=None, weights=None, output='res'):
    # flexible anti-crossing model, unique data. Only considering data
    
    f_fixed = params['f_fixed'].value
    g_hz = params['g_hz'].value
    
    model = anticrossing_mod2(f_tune, f_fixed, g_hz, data)
    # Calculate residuals and apply weights if provided
    if output == 'res':
        model = anticrossing_mod2(f_tune, f_fixed, g_hz, data)
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        return residuals
    return model

"""---Anticrossing mod3 and res3 - good in minimizing multiple upper and
lower branchs of anti-crossing but bad in sequential demonstration in anti-crossing
preferred model for fitting multiple branchs with each upper and lower branches
in the same frequency-------------------------------------------------------"""

def anticrossing_mod3(f_tune, f_fixed, g_hz, branch='up'):
    # ver3 -> replace y-data for ref with branch labels for cleaner release.
    """
    Solving the issue of upper and lower branches with the same frequency.
    Enables 1-1 mapping between upper and lower branches in one array.
    """
    A = (f_fixed+f_tune)/2
    B = np.sqrt(np.power(g_hz,2)+ np.power((f_fixed-f_tune)/2,2))
    f_plus, f_minus = A + B, A - B
    # Find the closest branch to each y_data point
    return np.where(branch == "up", f_plus, f_minus) # Match structure of y_data

def anticrossing_res3(params, f_tune, data=None, branch=None, weights=None, output='res'):
    """
    Fit model that accommodates upper and lower branches at the same frequency
    but having up and down frequencies.
    
    Data must have "up" and "down" branches for the plots
    
    # this is the official anti-crossing function that uses lmfit.Minimizer
    due to the non-continuous nature of the upper and lower branches of the
    minimizer
    
    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    f_tune : 1D numpy array
        DESCRIPTION.
    data : bool or 1D numpy array, optional
        ydata. The default is None.
    branch_lbl : bool or 1D numpy array, optional
        branch label == 'upper' -> gets upper branch
        branch label == 'lower' -> gets lower branch
    weights : 1D numpy array, optional
        DESCRIPTION. The default is None.
    output : string, optional
        res. The default is res.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    f_fixed = params['f_fixed'].value
    g_hz = params['g_hz'].value
    
    # model represents an array 
    model = anticrossing_mod3(f_tune, f_fixed, g_hz, branch)
    # Calculate residuals and apply weights if provided
    if output == 'res':
        model = anticrossing_mod3(f_tune, f_fixed, g_hz, branch)
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        return residuals
    # if output != 'res', then plot the modelled value.
    return model
