# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:56:27 2025

lineshapes_volt_to_freq

Voltage to Frequency / Coupling and vice versa Strength Conversion
    1. Find voltage to frequency conversion of qubit frequency Qj / Cj
        1.1. Ejmax = Ej1 + Ej2
        1.2. d = Ej1 - Ej2 / Ejmax
        1.3. A_conv (related to Inductive Coupling of Transmission line)
        1.4. phi_0
        1.5. Ec
        1.6. B = (1-d^2)/2, C = (1+d^2)/2 => B + C = 1; C = 1 - B 
        => 1 - (1+d^2)/ 2 => B => -0.5 + d^2/2
    2. Find coupling strengths g_{qj,cj}, g_{qj,cj+1} and qubit frequency of
        Cj via 3 spectras (qubit) from Qj+1 spectroscopy
        2.1. 2D spectra of Probability of Qj in |1> for ZPA of coupler and 
        Qj + 1 frequency => far-off-resonance => Cj frequency fit
        2.2. 2D spectra of Probability of Qj in |1> for ZPA coupler vs Qj + 1
        g_{qjcj} anticrossing
        2.3. 2D spectra of Probability of Qj in |1> for ZPA coupler vs Qj + 1
        g_{qjcj+1} anticrossing
    3. Find effective coupling strength due to coupler frequency
        3.1. via swap time (bridge) - ZPA of coupler vs Rabi time as functions of probability in |1>
        3.2. Effective coupling vs Coupler frequency

Prerequisite for frequency calibration:
    1. fixed flux distortion
    2. Applied flux matrix cancellation to minimize flux crosstalk    

@author: Mai
"""

import numpy as np
#import qutip as qt

from scipy.special import mathieu_a, mathieu_b
from scipy.optimize import minimize
#from qutip import basis

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
    f01 = np.sqrt(8*f_ej*f_ec)-f_ec
    (f01+f_ec)^2 = 8*f_ej*f_ec => f_ej= (f01+f_ec)^2/(8*f_ec)

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
    return np.power(f01+fc,2)/(8*fc) #previously f01+fc

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

    # Compute eigenvalues using Mathieu functions
    # eigenvalues = []
    # for n in range(m):
    #     if n % 2 == 0:
    #         # Even solutions: Mathieu characteristic values (a_n)
    #         eigenvalue = mathieu_a(n, q) * fc
    #     else:
    #         # Odd solutions: Mathieu characteristic values (b_n)
    #         eigenvalue = mathieu_b(n, q) * fc
    #     eigenvalues.append(eigenvalue)
    
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

def d_from_area(RJ, Area_1, Area_2):
    # converting normal state resistance to normal state resistivity / Area*Area
    return 

def fact_to_d(fact):
    # fact = Ej2/Ej1
    return (fact-1)/(fact+1)

def d_to_fact(d):
    """
    Convert Literature d to fact=Ej2/Ej1
    """
    return (1+d)/(1-d)

def Ej1_Ej2_d(Ejmax, gam):
    """
    Extract Ej1, Ej2 and d from the split transmon
    Note: gam = fact = Ej2/Ej1
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
    """Based on Josephson Inductance. Conversion of Mutual Inductance
    from sensitivity of flux-line to drive voltage (A_conv) to flux quanta"""
    return phi0*A_conv*Z0

def volts_to_norm_flux(vpk, A_conv, v0):
    """
    Conversion from applied voltage to normalized flux quanta
        norm_flux = np.pi*(phi_ext-phi_0)/phi_0 = A_conv*(vpk-v0)
        
    To avoid small values and improve readability, we just normalized flux in 
    units of product of pi and flux quanta (phi_0). 
    
    Use cases:
        1. secondary axis to recast voltage to flux quanta
        2. manuscript talking points (remember to multiply factor by pi if
                                      we talk about numerics)
    
    Parameters
    ----------
    vpk : 1D numpy array
        applied flux bias (V).
    A_conv : float
        conversion related to flux line mutual inductance.
    v0 : 1D numpy array
        flux offset (V).

    Returns
    -------
    1D numpy array
        units in normalized flux (phi_0).
    """
    return A_conv*(vpk-v0)

def norm_flux_to_volts(vpk, A_conv, v0):
    """
    confirmed to be inverse
    Use cases:
        1. secondary axis to express flux quanta (in theory) to voltage
        2. manuscript talking points (useful in guessing AWG flux bias voltage
                                      from theory
    
    
    Parameters
    ----------
    vpk : 1D numpy array
        External flux quanta(note that we used it for inverse func only).
    phi_conv : float
        conversion related to flux line mutual inductance.
    v0 : 1D numpy array
        flux offset (in voltage).

    Returns
    -------
    1D numpy array
        units in normalized flux (phi_0).
    """
    phi_conv = 1/A_conv
    return phi_conv*(vpk)+v0

"""much tangible transmon parameter could be obtained by getting f_max rather than f_Ej
and then get the f_Ej separately in the analysis. In this way, Ejmax do not need
some guesswork since they can be determined visually. In the same way, f_min
determines f_d."""

def fq_volts_to_hz(vpk, A_conv, v0=0, f_Ejmax=2*50*200E6, d=1, f_Ec=200E6, m=1, full=True):
    """
    Parabolic Approximation for qubit and Coupler. For f01 state
    use cases:
            1: single-qubit voltage to frequency conversion for flux pulses, if Ej is input
    
    Verdict (2025/03/18 => not preferred as f_Ejmax needs to be extracted
             for a better fit. Rather, we use f_max which is a measurable 
             component)
    
    Parameters
    ----------
    vpk : TYPE
        DESCRIPTION.
    A_conv : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    Ejmax : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.
    Ec : TYPE
        DESCRIPTION.
    m : int
        Energy levels: 1 -> 01
        2 -> 01, 12, etc
    full : boolean
        True, 7-parameter equation
        False, four-parameter equation
    
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # calculation, use frequency units
    
    if full==False:
        """parabolic approximation for voltage to frequency scaling"""
        A = np.sqrt(np.power(d*A_conv*(vpk+v0),2) + 1)
    else: 
        """full model for parametric voltage to frequency scaling, 
        assuming periodic shown"""
        phase = 2*A_conv*(vpk+v0) 
        B = 0.5*(1 + d**2)
        C = 0.5*(1 - d**2)
        A = np.sqrt(B+C*np.cos(phase))
    
    result = np.sqrt(8*f_Ejmax*f_Ec*A) - f_Ec
    return result #result/h

def fq_hz_to_volt(f, A_conv, v0=0, f_Ejmax=2*50*200E6, d=1, f_Ec=200E6, full=True):
    
    """useful for qubit / coupler voltage for mapping, useful for identifying 
    operation point for iSWAP and CZ gate.
    
    Verdict (2025/03/18 => not preferred as f_Ejmax needs to be extracted
             for a better fit. Rather, we use f_max which is a measurable 
             component)
    """
    
    # note, ej is input and defined d., useful in literature work
    
    # this is for conversion purposes
    
    params = np.power((h*(f-f_Ec)),4)/np.power(8*h*f_Ejmax*h*f_Ec,2)
    if full==False: 
        """parabolic approximation for frequency to voltage scaling.
        cannot assume d=0 otherwise the values blow up."""
        A = params/np.power(d,2)
        D = np.sqrt(A-1)
        result = (D - A_conv*v0)/A_conv
    else: 
        """full model for parametric frequency to voltage scaling, 
        assuming periodic shown. no issue of numeric blow-up if d=0
        to guess if d value, one can inspect parabolic effects of fc,
        fq_min, fq_max, useful for non-symmetric transmon.
        """
        B = 0.5*(1 + d**2)
        C = 0.5*(1 - d**2)
        D = np.arccos((params-B)/C)
        result = (D - 2*A_conv*v0)/(2*A_conv)
    return result

def f_coup_volt_to_hz(vpk, A_conv, f_max, f_Ec):
    # this model absorbs phase offset due to stray flux and d via Ej,eff, removing two unknowns.
    
    # the method assumes a cosine behavior, whose behavior is absorbed by a quartic cosine function.
    result = (f_max + f_Ec)*np.power(np.cos(2*A_conv*vpk), 0.25) - f_Ec
    
    return result

def f_coup_hz_to_volt(f, A_conv, f_max, f_Ec):
    
    result = np.arccos(np.power((f_max + f_Ec)/(f + f_Ec),4)/(2*A_conv))
    
    return result

# estimate Ej from mathieu's function for better accuracy

# for qutip simulation, useful to obtain bare state from dressed states.

def eigen_qcq_system(t, args_qcq):
    
    # eigenenergies of qj,cj and qn system, get 0 and 1 (focused on w01 and w12 frequencies)
    
    # assume coupler d=0, assume g_qjqn is negligible in this design
    
    # setting iSWAP and CZ. Enable bare and dressed states
    
    # setting up of variables
    w_qj = 2*np.pi*args_qcq['f_qj'] # idle-frequency
    w_qn = 2*np.pi*args_qcq['f_qn'] # idle-frequency
    w_qc = 2*np.pi*args_qcq['f_c'] # sweet-spot frequency
    
    w_Ec_qj = 2*np.pi*args_qcq['Ec_qj'] # Ec frequency
    w_Ec_qn = 2*np.pi*args_qcq['Ec_qn'] # Ec frequency
    w_Ec_c = 2*np.pi*args_qcq['Ec_c'] # Ec frequency
    
    g_qjcj = 2*np.pi*args_qcq['g_qjcj'] # coupling strength of qj with 
    g_qncj = 2*np.pi*args_qcq['g_qncj'] # coupling strength of qj with ej
    g_qjqn = 2*np.pi*args_qcq['g_qjqn'] # coupling strength of qj with ej
    
    dim = 2  # level number for each qubit or coupler = 0~(dim-1)
    # dim=2; function is for ISWAP and dressed state frequency
    # dim=3; function is for CZ frequency
    dim_tot = dim**3
    
    j = np.arange(dim) #np #dr wang's approximation is false.
    
    """
    For coupler control, better calibrate system according to EJJ, Aconv, d
    and Vc offset of coupler. Better use Vcc offset for analysis.
    """
    # set bare-state eigenvalues
    
    return 

def dressed_eigen_q_cj(f_c, f_q, g_qc_hz):
    """Show dressed eigensystem spectra for strong coupling between 1 moving
    qubit f_c and a fixed frequency coupler"""
    
    # fit to coupled qj,cj system and qj+1,cj system
    # Nearest neighbor q_nn is q_j or q_{j+1}, coupler is cj
    A = 0.5*(f_q+f_c)    
    B = np.sqrt(np.power(g_qc_hz,2) + np.power(f_q-f_c,2)/4)
    f_qc_p = A + B
    f_qc_m = A - B
    return f_qc_p, f_qc_m


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

def f01_idle(vpk, f_idle):
    """
    20250318 - Output idle frequency
    
    Use cases:
        1. Plotting bare-state of idle frequencies (rather than using qutip for
                                                    diagonalized bare-states)
         (less dependencies)

    Parameters
    ----------
    vpk : 1D numpy array
        idle frequencies.
    f_idle : float
        Idle frequencies in Hz or GHz.

    Returns
    -------
    1D numpy array.

    """
    arr = np.ones(len(vpk))
    return f_idle*arr

def f01_hz_to_volts(f01, A_conv, v0=0, f_max=5E9, fact=1, f_Ec=200E6):
    """
    short-hand method on determining best corresponding voltage for appropriate bias frequency
    
    np.sqrt(f_Ejmax*f_ec)=f_max+f_ec
    
    note that output will be positive voltage. For negative voltage, take the negative sign.
    
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

"""---------------------------Ver2 model--obsolete-------------------------"""

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

"""---------------------Ver3 model-most-useful model -----------------------"""
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

"""---further analysis---extracting voltages where bare-state intersects----"""


"""--------------------Work it out------------------------------------------"""


"""----Coupling strength fits - bridge methods, reference = f01_max---------"""
def P_01_g(t, g_tilde_hz, f_qj_, f_qn_):
    """
    Modelling swap interaction in transmon bridges (no decoherence / damping)
    - best fit function for extracting g_tilde would be using 
        cosine-fit with offset. But in principle, best fit function lies
        with good state preparation (if iSWAP would be a gate interaction)
    
    - reference based on equal g and delta
    Parameters
    ----------
    t : 1D numpy array
        Pulse duration (s, ns).
    g_tilde_hz : float / 1D numpy array
        effective coupling strength (in Hz, GHz).
    f_qj__ : float
        dressed qubit frequency (in Hz, GHz).
    f_qn__ : float
        dressed neighbor qubit frequency (in Hz, GHz).

    Returns
    -------
    1D numpy array / 2D numpy array
        |01> swap
    """
    # initialization:
    g__ = 2*np.pi*g_tilde_hz
    w_qj = 2*np.pi*f_qj_
    w_qn = 2*np.pi*f_qn_
    
    add = np.sqrt(np.power(g__,2) + np.power(w_qj-w_qn,2))
    if add < 0:
        fourier = - np.abs(add)
    else: 
        fourier = np.abs(add)
    return 0.5*np.cos(fourier*t)+0.5

def g_qq_volt_to_hz(vc, g_qjqn_hz, g_qjcj_hz, g_qncj_hz, f, kwargs_fc):
    """
    Effective coupling strength mediated by coupler.
    # reference has wrong units. (rather than w_cj-w, it should be w_cj + wj)
    Parameters
    ----------
    vc : TYPE
        DESCRIPTION.
    g_qjqn_hz : TYPE
        DESCRIPTION.
    g_qjcj_hz : TYPE
        DESCRIPTION.
    g_qncj_hz : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    kwargs_fc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # effective coupling between qubit and coupler
    
    # declaration of variables for coupler frequency
    g_qjqn = 2*np.pi*g_qjqn_hz # free parameter
    g_qjcj = 2*np.pi*g_qjcj_hz # this should be constant
    g_qncj = 2*np.pi*g_qncj_hz # this should be constant
    w = 2*np.pi*f # f = f_qj = f_qn
    
    # coupler frequency
    w_cj = 2*np.pi*f01_volts_to_hz(vpk=vc, A_conv=kwargs_fc['A_conv'], 
                                   v0=kwargs_fc['v0'], 
                                   f_max=kwargs_fc['f_max'], 
                                   fact=kwargs_fc['fact'], f_Ec=kwargs_fc['f_Ec'])
    
    # expression, assuming wcj != w, peak
    
    """
    Liang, G.-H., ... Zheng, D.-N. (2023). Tunable-coupling architectures with 
    capacitively connecting pads for large-scale superconducting multiqubit 
    processors. Physical Review Applied, 20(4), 044028. 
    """
    res = g_qjqn - (g_qjcj*g_qncj)/(1/(w_cj - w) + 1/(w_cj + w))
    
    # # Ensure no division by zero
    # if np.isscalar(w_cj):
    #     return g_qjqn /(2*np.pi) if w_cj == w else res/(2*np.pi)
    # else:
    #     result = np.where(w_cj == w, g_qjqn, res)
    #     return result/(2*np.pi)
    
    # basic assumption.
    return res/(2*np.pi)

def czbr_freq_to_geff(freq_hz, det_hz):
    """
    Eqn: freq = np.sqrt(4*g_eff^2 + det^2)
    freq^2 = 4*g_eff^2 + det^2
    g_eff = np.sqrt((freq^2 - det^2)/4)
    
    g_eff = g01_swap

    Parameters
    ----------
    freq_hz : 1D numpy array
        Frequency extracted from the P_11 probability vs time at varying coupler
        voltage.
    det_hz : float
        detuning between qj_01 freq and qn_12 freq.

    Returns
    -------
    geff : 1D numpy array
        Effective Coupling Strength in hz.
    """
    freq_sqr = np.power(freq_hz,2) # period
    det_sqr = np.power(det_hz,2) # detuning squared
    g_sqr = (freq_sqr - det_sqr)/4
    # v1
    g_arr = np.where(freq_sqr >= det_sqr, np.sqrt(np.abs(g_sqr)), 
                     np.sqrt(np.abs(g_sqr)))
    # v2
    # add = np.sqrt(eqn_sqr)
    # eqn_arr = np.where(add < 0, -1*np.abs(add), np.abs(add))
    
    g_arr = np.where(freq_sqr >= det_sqr, np.sqrt(np.abs(g_sqr)), 
                     np.sqrt(np.abs(g_sqr)))
    return g_arr

def signed_cos(x, freq, phase, amp, ofs):
    return amp*np.cos(freq*x + phase) + ofs

def g_qq_hz1(fc, gic_hz, K, f1, tol=1E4, max_amp=400E6, gate='iswap'):
    """Approximate equation for determining g1cg2c and g_qq via empirical
    fits. Not meant to be accurate but to provide good guess function for
    linear fits. Simplified without zz_g and detuning.
    
    fc => coupler frequency (Hz) (1D numpy array)
    g_ic => qubit coupler strength
    K => scaling factor for g_qq (relative to g_qq)
    f1 => operation point
    gate => 'iswap' or 'cz'
    """   
    # Prevent division by zero near singularities
    # fc = np.where(np.isclose(fc,  f1, atol=tol),  f1 + tol, fc)
    # fc = np.where(np.isclose(fc, -f1, atol=tol), -f1 + tol, fc)
    # 20250516 - ValueError: operands could not be broadcast together with shapes (728,) (2,) 
    
    g_ic = 2*np.pi*gic_hz
    g_qq = 2*np.pi*(gic_hz/K)
    wc = 2*np.pi*fc
    w1 = 2*np.pi*f1
    
    if gate=='iswap':
        const=1
    else:
        const=np.sqrt(2)
    
    # compute function
    g = const*(g_qq - 2*np.power(g_ic,2)*wc/(np.power(wc,2)-np.power(w1,2))) 
    
    # perform clipping
    return np.clip(g/(2*np.pi), -max_amp, max_amp) # working

def g_qq_hz_res1(params, fc, data=None, weights=None, gate='iswap', tol=1E4, 
                 max_amp=400E6, output='res'):
    """
    20250515
    Residual fit for g_swap and g_cz for coupler because the fc 
    is parabolic. This is to simplify minimization prior to more complicated
    minimization practices like adjusting qubit-bias to frequency => subject
    to more complications
    
    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    fc : 1D numpy array
        Coupler tuning in Hz (parabolic). Nonlinear
    data : bool or 1D numpy array, optional
        ydata. The default is None.
    weights : 1D numpy array, optional
        DESCRIPTION. The default is None.
    gate : float
        'iswap' => correction factor of 1
        'cz' => correction factor of sqrt(2)
    output : string, optional
        res. The default is res => residual.
        mod. The output is the model function

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # initiate parameters    
    g_qjcj = params['gic_hz'].value # fit function
    K = params['K'].value # fit function
    f1 = params['f1_hz'].value
    
    # model represents an array 
    model = g_qq_hz1(fc, g_qjcj, K, f1, tol, max_amp, gate)
    # Calculate residuals and apply weights if provided
    if output == 'res':
        model = g_qq_hz1(fc, g_qjcj, K, f1, tol, max_amp, gate)
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        # Apply a soft penalty where model > MAX_AMPLITUDE
        penalty = np.where(model > max_amp, (model - max_amp) ** 2, 0)
        return residuals + penalty  # Penalized residual
    # if output != 'res', then plot the modelled value.
    return model

def g_qq_hz2(fc, gic_hz, K, zz_hz, f1, det_hz, tol=1E4, max_amp=400E6, gate='iswap'):
    """Alternative model for comprehensive coupling strength mediated by
    coupling in the case of detuning. zz_hz is included along with gg_coupling
        
    fc => coupler frequency (Hz) (1D numpy array)
    g_ic => qubit coupler strength
    K => scaling factor for g_qq (relative to g_qq)
    f1 => operation point
    det_hz
    gate => 'iswap' or 'cz'
    """
    f2 = f1 + det_hz
    
    # Prevent division by zero near singularities
    # fc = np.where(np.isclose(fc, f1, atol=tol), f1 + tol, fc)
    # fc = np.where(np.isclose(fc, -f1, atol=tol), -f1 + tol, fc)
    # fc = np.where(np.isclose(fc, f2, atol=tol), f2 + tol, fc)
    # fc = np.where(np.isclose(fc, -f2, atol=tol), -f2 + tol, fc)
       
    # declair values
    wc = 2*np.pi*fc # x-axis    
    g_ic = 2*np.pi*gic_hz # fit
    g_qq = 2*np.pi*(gic_hz/K) # derived value
    w1 = 2*np.pi*f1
    w2 = 2*np.pi*f2
    zz = 2*np.pi*zz_hz
    
    # correction constant for fit    
    if gate=='iswap':
        const=1
    else:
        const=np.sqrt(2)
    
    # compute function
    B = wc*(1/(np.power(wc,2)-np.power(w1,2))+1/(np.power(wc,2)-np.power(w2,2)))
    g = const*(g_qq - np.power(g_ic,2)*B)
    # add zz effects
    g_mod = np.where(g>=0, g + const*zz, g - const*zz)
    # g_mod = g
    # perform clipping
    return np.clip(g_mod/(2*np.pi), -max_amp, max_amp) # working

def g_qq_hz_res2(params, fc, data=None, weights=None, gate='iswap', tol=1E4, 
                 max_amp=400E6, output='res'):
    """
    20250516 - inclusion of complicated effects
    Residual fit for g_swap and g_cz for coupler because the fc 
    is parabolic. This is to simplify minimization prior to more complicated
    minimization practices like adjusting qubit-bias to frequency => subject
    to more complications
    
    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    fc : 1D numpy array
        Coupler tuning in Hz (parabolic). Nonlinear
    data : bool or 1D numpy array, optional
        ydata. The default is None.
    weights : 1D numpy array, optional
        DESCRIPTION. The default is None.
    gate : float
        'iswap' => correction factor of 1
        'cz' => correction factor of sqrt(2)
    output : string, optional
        res. The default is res => residual.
        mod. The output is the model function

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # initiate parameters    
    g_qjcj = params['gic_hz'].value # fit function
    K = params['K'].value # fit function
    zz_hz = params['zz_hz'].value
    f1 = params['f1_hz'].value
    det = params['det_hz'].value
    
    # model represents an array 
    model = g_qq_hz2(fc, g_qjcj, K, zz_hz, f1, det, tol, max_amp, gate)
    # Calculate residuals and apply weights if provided
    if output == 'res':
        model = g_qq_hz2(fc, g_qjcj, K, zz_hz, f1, det, tol, max_amp, gate)
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        # Apply a soft penalty where model > MAX_AMPLITUDE
        penalty = np.where(model > max_amp, (model - max_amp) ** 2, 0)
        return residuals + penalty  # Penalized residual
    # if output != 'res', then plot the modelled value.
    return model

def flip_sign_mod(xdata, ydata, dict_fc, dict_fit, **kwargs):
    # process sign of y-data according to positive frequency fit
    x_fc = f01_volts_to_hz(vpk=xdata, A_conv=dict_fc['A_conv'],
                           v0=dict_fc['v0'], f_max=dict_fc['f_max'],
                           fact=dict_fc['fact'], f_Ec=dict_fc['f_Ec'])
    
    if 'zz_hz' and 'det_hz' in dict_fit:
        y_mod = g_qq_hz2(fc=x_fc,gic_hz=dict_fit['gic_hz'], K=dict_fit['K'],
                         f1=dict_fit['f1_hz'], zz_hz=dict_fit['zz_hz'], 
                         det_hz=dict_fit['det_hz'])
        print('zz') # use zz_points to make the inflection points (assuming range is okay)
        # make the mask where ydata >=0
        indices = np.arange(len(ydata))
        mask = ydata>=0
        x_fcp = x_fc[mask]
        y_prox = ydata[mask]
        idx_fcm = np.where(x_fcp == np.amax(x_fcp))[0][0]
        print(idx_fcm)
        if 'zz_l' in kwargs:
            zz_l = kwargs['zz_l']
        else:
            zz_l = np.amin(np.abs(y_prox[:idx_fcm])) # need index for positive or negative
        if 'zz_r' in kwargs:
            zz_r = kwargs['zz_r']
        else:
            zz_r = np.amin(np.abs(y_prox[idx_fcm:])) # need index for positive or negative
        #zz_r = np.amin(np.abs(ydata[idx_fcm:]) and ydata>=0)
        #print(zz_r)
        # use obtained zz_l and zz_r to find the accepted minimum indices
        idx_l = np.where(np.abs(ydata)==zz_l)[0][0]
        idx_r = np.where(np.abs(ydata)==zz_r)[0][0]
        print(idx_l)
        print(idx_r) # this got 893, not 792. Should be robust.
        # using numpy masking for identification.
        y_process = np.zeros(len(ydata), dtype=float)
        
        # Boolean masks
        in_range = (indices >= idx_l) & (indices <= idx_r)
        out_of_range = ~in_range
        y_process[in_range] = np.abs(ydata[in_range])
        y_process[out_of_range] = -np.abs(ydata[out_of_range])
    else:
        y_mod = g_qq_hz1(fc=x_fc,gic_hz=dict_fit['gic_hz'], K=dict_fit['K'],
                         f1=dict_fit['f1_hz'])
        y_process = np.where(y_mod>=0, ydata, -ydata)
        print('norm')
    # first idea, y_mod = 0
    # y_process = np.where(y_mod>=0, ydata, -ydata)
    
    # second idea, identify idx_zz_l and idx_zz_r within a certain range and get min
    # the flip-sign works well in accordance to the fit model.
    return y_process

"""------------------disable gg_qq_hz too-----------------------------------"""

def g_qq_hz(fc, g_qjqn_hz, g_qjcj_hz, zz_hz, f1_hz, det_hz, max_amp=200E6, 
            tol=1E3, gate='iswap'):
    """
    # best to use g_qjcj_hz = np.sqrt(g_qjcj_hz*g_qj+1cj_hz)
    
    effective coupling strength formulation for iSWAP and CZ
    Comparing iSWAP and CZ formulation:
        1. iSWAP, det_hz = f2_10 - f1_10 => f2_10 = f1_10+ det_hz
        2. CZ, det_hz = f2_21 - f1_10 => (f2_10 - f2_Ec) - f1_10
        Cannot discount zz_hz which is the effective baseline
    
    fit parameters, g_qjqn_hz, g_prod_hz (geometric mean in hz, assuming symmetric), zz_hz
    non_tuning parameters: f1_hz, det, g_prod_hz (in principle but we are 
                                                  missing data)
    
    """
    
    # declaration of variables for coupler frequency
    g_qjqn = 2*np.pi*g_qjqn_hz # free parameter
    g2_prod = np.power(2*np.pi*g_qjcj_hz,2) # not assuming geometric product
    zz = 2*np.pi*zz_hz # zz-interaction frequency, avoiding coupling to be off
    w1 = 2*np.pi*f1_hz # q1
    det = 2*np.pi*det_hz # detuning of 2 with respect to q1
    w_cj = 2*np.pi*fc
    w2 = w1 + det
    # Dong-Ning Zheng
    
    # frac / assume non-detuned data
    frac1 = (1/(w_cj - (w1+tol))) + (1/(w_cj + (w1+tol)))
    # frac1
    frac2 = (1/(w_cj - (w2+tol))) + (1/(w_cj + (w2+tol)))
    # frac2
    """If wc2=wc1, frac_sum = 2*frac_1"""
    
    # eqn, 
    res = (g_qjqn - (g2_prod/2)*(frac1 + frac2))
    # zz avoids the coupling seriously
    res_mod = np.where(res>=0, res+zz, res-zz)
    # res_mod = res
    if gate=='cz':
        eqn= np.sqrt(2)*res_mod/(2*np.pi)
    else:
        # gate is iswap
        eqn= res_mod/(2*np.pi)
        
    return np.clip(eqn, -max_amp, max_amp) # working

def g_qq_hz_res(params, fc, data=None, weights=None, gate='iswap', tol=1E3, 
                 max_amp=2E5, output='res'):
    """
    Residual fit for g_swap and g_cz for coupler because the fc 
    is parabolic. This is to simplify minimization prior to more complicated
    minimization practices like adjusting qubit-bias to frequency => subject
    to more complications
    
    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    fc : 1D numpy array
        Coupler tuning in Hz (parabolic).
    data : bool or 1D numpy array, optional
        ydata. The default is None.
    weights : 1D numpy array, optional
        DESCRIPTION. The default is None.
    output : string, optional
        res. The default is res.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    g_qjqn = params['g_qjqn_hz'].value # fit function
    g_qjcj = params['g_qjcj_hz'].value # fit function
    zz = params['zz_hz'].value
    f1 = params['f1_hz'].value
    det = params['det_hz'].value
    
    # model represents an array 
    model = g_qq_hz(fc, g_qjqn, g_qjcj, zz, f1, det, tol, 
                    max_amp, gate=gate)
    # Calculate residuals and apply weights if provided
    if output == 'res':
        model = g_qq_hz(fc, g_qjqn, g_qjcj, zz, f1, det, tol, max_amp, 
                        gate=gate)
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        return residuals
    # if output != 'res', then plot the modelled value.
    return model

def xtalk_g_qq_hz_to_hz(fc, mxtalk, g_qjqn_hz, g_prod_hz2, f):
    """
    similar design as g_fc but modelled with xtalk: determined
    after precisely determined idle frequency
    
    mxtalk in - of slope.
    """
    # declaration of variables for coupler frequency
    g_qjqn = 2*np.pi*g_qjqn_hz # free parameter
    g2_prod = np.power(2*np.pi*g_prod_hz2,2)
    w = 2*np.pi*((-mxtalk)*np.ones(len(fc))+f)
    w_cj = 2*np.pi*fc
    
    # Dong-Ning Zheng
    res = g_qjqn - g2_prod/(1/(w_cj - w) + 1/(w_cj + w))
    return res/(2*np.pi)

def g_qq_hz_to_volts(g_tilde_hz, g_qjqn_hz, g_qjcj_hz, g_qncj_hz, f, kwargs_fc):
    
    """
    20250327 - notes
    has to do correction due to wrong units of wcj - wj (Y. Shi's derivation
    in supplement was wrong due to sign use. D-N. Zheng's derivation is
    better written, and corresponds with data.
    
    Note that analytic description assumes (1/(wj + wcj) << 0) for short
    wcj range. But this results in avoiding resonance descriptions between
    w_cj and w_j, which results in blow-up when wcj = wj. This apprximation 
    is only valid at not-so high resonance points.
    
    Solving the original solution would require root-finding approaches
    to enable resolution.
    """
    
    # effective voltage needed to control coupler for coupling (after fit)
    
    # ver1, with phase offset
    
    # initiate variables due to tedius paramaters of coupler
    g_tilde = 2*np.pi*g_tilde_hz
    g_qjqn = 2*np.pi*g_qjqn_hz #g_qjq(j+1)
    g_qjcj = 2*np.pi*g_qjcj_hz
    g_qncj = 2*np.pi*g_qncj_hz#g_q(j+1)cj
    w = 2*np.pi*f # coupling between q1 and q2
    
    # parameters of coupler (in f_max and Ej manner for simplicity, Ej will be later)
    w_Ecc = 2*np.pi*kwargs_fc['f_Ec']
    w_maxc = 2*np.pi*kwargs_fc['f_max']
    v0_c = kwargs_fc['v0']
    Ac = kwargs_fc['A_conv']
    fact_c = kwargs_fc['fact'] # units in np.pi/Vc
    dc = (fact_c-1)/(fact_c+1)
    
    # equation for fit.
    g_val = w-g_qjcj*g_qncj/(g_tilde-g_qjqn) # correction, wrong from S1 of Heng Fan.
    var1 = np.power((g_val + w_Ecc)/(w_maxc+w_Ecc),4)
    Bc = (1 + dc**2)
    Cc = (1 - dc**2)
    Dc = 1/(2*Ac)
    
    # use equation (simplified)
    var2 = Dc*np.arccos((2*var1-Bc)/Cc)     
    g_volt_n = v0_c - var2 # in volts
    g_volt_p = v0_c + var2 # in volts
    
    return [g_volt_n, g_volt_p]

def P_11_hel(t, det_hz, g_hz):
    """
    Approximate Probability of |11> <-> |20> interaction
    Christoph Hellings, ..., Andreas Wallraff. arXiv:2503.04610v1
    
    Approximate detuning as function of applied pulse amplitude
    (1st-order taylor approximation around point of zero
     detuning)
    
    Check to map for comparison and optimization of the 
    CZ leakage. (1-P|11>) in Y.-K. Sung experiment. If this
    model works for 2D architecture, then we use them.
    
    20250502 -> assessment of the modelling, g_eff_hz and det_hz must be
    scaled by factor of 4 to match. - seems that qualitatively, the model
    by Anferov on Rabi is much better.
    
    Parameters
    ----------
    t : float / 1D numpy array
        Interaction time.
    det_hz : float / 1D numpy array
        Detuning between |11> and |20> states.
    g_eff_hz : float / 1D numpy array
        Coupling strength between |11> and |20> / |02> states.

    Returns
    -------
    None.

    """
    # initialization
    det = 2*np.pi*det_hz
    g = 2*np.pi*g_hz
    
    # equation
    res_A = np.power(2*g,2) + np.power(det,2)
    #res_B = 0.5*np.power(2*g,2) + np.power(det,2)*0.5
    res_B = 0.5*np.power(2*g,2) + np.power(det,2)*0
    eqn = res_B*(1+np.cos(t*np.sqrt(res_A)))/res_A
    return eqn

def P_11_hel1(t, det_hz, g_hz, reduce=1):
    """
    20250502 -> assessment of the modelling, g_eff_hz and det_hz must follow 
    notation by sin^{2}
    
    based on sin^2(x) = 0.5*(1-cos(2x)), fsim definition. => rabi-oscillation
    explicitly mentioned by fsim 
    - conflicting model by Hellings, and Y. Shi (Heng-fan group). Most consistent
    definition may come from foxen and martinis but without elaboration.
    - reported by Fumiki Yoshinaga and J. Tsai that Rabi Oscillation =>
    Physical Review B, 89(2). (omega_Rabi = 2*g)
    - correct solution. Problem is the frequency.
    
    Parameters
    ----------
    t : float / 1D numpy array
        Interaction time.
    det_hz : float / 1D numpy array
        Detuning between |11> and |20> states.
    g_eff_hz : float / 1D numpy array
        Coupling strength between |11> and |20> / |02> states.
    reduce : float
        Pulse area deduction factor due to rising and falling edge non-correction

    Returns
    -------
    None.

    """
    # initialization
    det = 2*np.pi*det_hz
    g = 2*np.pi*g_hz
    
    # equation
    res_A = np.power(2*g,2) + np.power(det,2) #eff_rabi^2 = (2g)^2+det^2
    #res_B = 0.5*np.power(2*g,2) + np.power(det,2)*0.5
    res_B = 0.5*np.power(2*g,2) + np.power(det,2)*0
    eqn = res_B*(1-np.cos(t*np.sqrt(res_A)*reduce))/res_A # based on sine
    return eqn