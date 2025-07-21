# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:15:29 2024

lineshapes - quantum computing - T1 Rabi, Ramsey, nTLS, RB, IRB, 

Programming Format based on Matt Neville:
    https://github.com/lmfit/lmfit-py/blob/master/lmfit/lineshapes.py

Lineshapes are independent of Matt Neville's basic functions but nevertheless
incorporated in some non-lmfit functions

@author: Mai
"""

import numpy as np

from uncertainties import ufloat, unumpy
from uncertainties.umath import sqrt

functions = ('exp_decay_c', 'exp_decay_cIQ', 'decay_cos_c', 'decay_cos_cIQ',
             'gaussian_exp_cfit', 'nlz_tls', 'rb_seq_fid', 'pb_seq_fid')

# fundamental constants
e0 = 8.8541878128E-12 #F m^s
mu0 = 1.25663706212E-16 #N A^-2
kb = 1.380649E-23 #J K^-1
hbar = 1.054571817E-34 #J s
c = 299792458 # speed of light m s^-2
phi0 = 2.067833848E-15 #Wb
h = 2*np.pi*hbar

def exp_decay_c(t, T0, Ioffs, Iamp):
    """
    exponential decay with constant_c for real-valued arrays
    Use cases:
        1. Rabi decay
        2. Exponential time constant
    
    Parameters
    ----------
    t : 1D numpy array
        time interval between drive and readout pulse.
    T0 : float
        characteristic decay time.
    Ioffs : float
        I plane offset (V or state probability).
    Iamp : float
        I amplitude (V or state probability.
    
    Returns
    -------
    1D numpy array of real numbers.

    """
    return Ioffs + Iamp*np.exp(-t/T0)

def exp_decay_cIQ(t, T0, Ioffs, Iamp, Qoffs, Qamp):
    """
    Complex-numbered exponetial decay with constants
    Use cases:
        1. Rabi decay
        2. Data-based time constants
    
    Parameters
    ----------
    t : 1D numpy array
        time interval between drive and readout pulse.
    Ioffs : float
        I plane offset (V or state probability).
    Iamp : float
        I amplitude (V or state probability.
    Qoffs : float
        Q plane offset (V or state probability).
    Qamp : float
        Q amplitude (V or state probability.
    T0 : float
        decay time.

    Returns
    -------
    1D numpy array of complex numbers.

    """
    #I = exp_decay_c(t=t, T0=T0, Ioffs=Ioffs, Iamp=Iamp)
    #Q = exp_decay_c(t=t, T0=T0, Ioffs=Qoffs, Iamp=Qamp)
    I = Ioffs + Iamp*np.exp(-t/T0)
    Q = Qoffs + Qamp*np.exp(-t/T0)
    return I + 1j*Q

def cosine(x, amplitude=1.0, frequency=1.0, shift=0.0):
    """Return a cosine function. #not noted - same notation as LMFIT

    cosine(x, amplitude, frequency, shift) =
        amplitude * sin(x*frequency + shift)

    """
    return amplitude*np.cos(x*frequency + shift)

def decay_cos_c(t, T0, fd, Ioffs, Iamp, phi_I):
    """
    general fitting model for decaying cosine function with y-offset.
    
    For Rabi oscillation, t the qubit drive pulse length, fd is the
    rabi drive frequency and T0 is the characteristic Rabi time.
    
    For Ramsey oscillation / fringes, t is the time duration between the +/-
    pi/2 pulse, 
    fd=(fdrive-fqubit) is the frequency detuning betweeen drive and 
    qubit frequency.
    T0 represents the dephasing time T2.

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
    return Ioffs + Iamp*np.cos(wd*t + phi_I)*np.exp(-(t/T0))

def decay_sin_c(t, T0, fd, Qoffs, Qamp, phi_Q):
    """
    general fitting model for decaying sine function with y-offset.
    Composite function do not work for the decay_sin_c
    
    For Rabi oscillation, t the qubit drive pulse length, fd is the
    rabi drive frequency and T0 is the characteristic Rabi time.
    
    For Ramsey oscillation / fringes, t is the time duration between the +/-
    pi/2 pulse, 
    fd=(fdrive-fqubit) is the frequency detuning betweeen drive and 
    qubit frequency.
    T0 represents the dephasing time T2.

    Parameters
    ----------
    t : 1D numpy array
        time depending on purpose.
    T0 : float
        characteristic time.
    fd : float
        drive frequency.
    Qoffs : float
        Offset value.
    Qamp : float
        Amplitude of oscillation.
    phi_I : float
        Phase offset in Radians.

    Returns
    -------
    1D numpy array.
    """
    wd = 2*np.pi*fd
    return Qoffs + Qamp*np.sin(wd*t + phi_Q)*np.exp(-(t/T0))
    
def decay_wave_cIQ(t, T0, fd, phi, Ioffs, Iamp, Qoffs, Qamp):
    """
    IQ decay time based on lmfit - longitudinal relaxation for T1 measurement
    Parameters
    ----------
    t : 1D numpy array
        time interval between drive and readout pulse.
    Ioffs : float
        I plane offset (V or state probability).
    Iamp : float
        I amplitude (V or state probability.
    Qoffs : float
        Q plane offset (V or state probability).
    Qamp : float
        Q amplitude (V or state probability.
    T0 : float
        decay time.

    Returns
    -------
    1D numpy array of complex numbers.

    """
    # define decaying sine function with shared T0, fd and phi_I
    # wd = 2*np.pi*fd
    I = decay_cos_c(t=t, T0=T0, fd=fd, Ioffs=Ioffs, Iamp=Iamp, phi_I=phi)
    #Q = Qoffs + Qamp*np.sin(wd*t + phi)*np.exp(-(t/T0)) #this is a sine function
    Q = decay_sin_c(t=t, T0=T0, fd=fd, Qoffs=Qoffs, Qamp=Qamp, phi_Q=phi)
    return I + 1j*Q

def vac_rabi_osc(t, T1, g, delta_wq, phi0):
    """
    Returns P_e(t) with decay

    Parameters
    ----------
    t : float
        time in s.
    T1 : float
        Decay time in s.
    g : float
        Rabi Coupling Rate with Cavity / Qubit in Hz.
    delta_wq : float
        Frequency detuning in seconds.
    phi0 : float
        Phase shift due to phase delay in radians.

    Returns
    -------
    1D numpy array
        Rabi-Rate.

    """
    # simulated rabi oscillations
    
    #initiate parameters
    g*=2*np.pi
    delta_wq*=2*np.pi
    omega = np.sqrt(np.power(2*g, 2)+np.power(delta_wq,2))
    amp = 2*np.power(g/omega, 2)
    offs = amp
    return (offs - amp*np.cos(omega*t+phi0))*np.exp(-t/T1)

def gaussian_exp_cfit(t, Tg, Ioffs, Iamp, T0):
    """
    Gaussian Decay due to spin-echo. Sensitive to low-frequency noise
    
    In LMFit, Composite Function is ConstantModel() + GaussianModel()*Exponential Model
    By Ramsey interferometry, it can be Gaussian-limited noise.
        
    Parameters
    ----------
    t : 1D numpy array
        time interval between two pulses.
    Tg : float
        fitting gaussian time errors.
    T0 : float
        Acquired T0 - constant.
    Ioffs : float
        Offset voltage.
    Iamp : float
        Measured Amplitude.

    Returns
    -------
    None.

    """
    return Ioffs + Iamp*np.exp(-t/(2*T0))*np.exp(-np.power(t/Tg, 2))

def nlz_tls(xdata, fi_arr, hwhm_tls_arr, g_tls_arr, hwhm_q):
    """
    Interacting TLS Lorentzian Model According to Barends et al 2013
    single TLS - function
    
    Replot TLS spectra according to multiple data for remodelling
    
    # we test the veracity of the function
    # should be flexible to multiple peaks
    
    According to Klimov 1/T1 = hwhm

    Parameters
    ----------
    xdata : 1D numpy array
        frequency in Hz.
    fi : float / numpy array
        TLS frequency in Hz.
    hwhm_tls : float / numpy array
        TLS linewidth in Hz (hwhm).
    g_tls : float / numpy array
        qubit-tls coupling in Hz.
    hwhm_q : float
        qubit linewidth in Hz (hwhm).

    Returns
    -------
    1D numpy array.
        TLS lorentzian in Hz
    """
    # initialize conditions for qubit linewidth baseline
    n_lz = len(fi_arr)
    gam_q = 2*np.pi*hwhm_q*np.ones(len(xdata))
    
    #initialize condition for list of tls parameters
    det_arr = [2*np.pi*(xdata - fi) for fi in fi_arr] # list of 1D numpy array
    gam_i_arr = [2*np.pi*hwhm_tls for hwhm_tls in hwhm_tls_arr]
    g_arr = [2*np.pi*g_tls for g_tls in g_tls_arr]
    
    #model peak response 
    resp_arr = np.ones((len(fi_arr), len(xdata)))    
    for i in range(n_lz):
        A = 2*np.power(g_arr[i],2)*gam_i_arr[i]
        B = np.power(gam_i_arr[i], 2) + np.power(det_arr[i], 2)
        resp_arr[i] = A/B
    
    # sum-all peaks in a 2D array
    summed_curve = np.sum(resp_arr, axis=0) + gam_q        
    return summed_curve/(2*np.pi)

def Gaussian(x, mu, sigma, Amp):
    # Set an adjustable gaussian distribution
    # for a normalized Gaussian Distribution, Amp=1
    A = 1/(sigma*np.sqrt(2*np.pi))
    B = np.exp(-(x-mu)**2/(2*(sigma**2)))
    return Amp*A*B

"""---------------Benchmarking Functions------------------------------------"""
def rb_seq_fid(m, A, p, B):
    """
    Average sequence fidelity for 1Q and 2Q randomized / interleaved benchmarking
    F(m)=A*p**m + B
    
    For ease in fitting, we use a composite model made up of
    C = PowerLawModel() + ConstantModel() from LMFIT package
    
    Parameters
    ----------
    m : 1D numpy array
        number of cliffords.
    A : float
        A - state preparation error.
    p : float
        1-p => rate of depolarization
    B : float
        state measurement error.

    Returns
    -------
    1D numpy array.

    """
    return A*(np.power(p, m)) + B

# for parametric benchmarking, we bundle data sequences
def pb_seq_fid(m, Ap, u, Bp):
    """
    Purity benchmarking for 1Q and 2Q. Useful for interleaved.
    P(m) =A'*u**(m-1) + B' except we put -1 in m 

    Parameters
    ----------
    m : 1D numpy array
        number of cliffords.
    A : float
        A - state preparation error.
    u : float
        unitarities
    B : float
        state measurement error.

    Returns
    -------
    1D numpy array.

    """
    return rb_seq_fid(m=m-1, A=Ap, p=u, B=Bp)

"""---------------for transmon function, must be in a class-----------------"""

"""Class for EjEc Relations"""
def freq_Ej(f01, Ec):
    """
    Determine Ej from f01 and anharmonicity
    input:
      f01 : float
      Resonant frequency in Hz
      Ec : float
      Charging Energy in Hz

    return
    """
    f_Ej = ((f01+Ec)/8)**2/Ec
    return f_Ej

def Ej_phi(phi, Ejmax, gam, phi_e):
    """
    Ej as a function of external flux phi
    input:
      phi : 1D numpy array
        external flux as a function of quanta
      Ejmax : float
        Total Ej in units of Hz
      d : float
        Junction asymmetry parameter
      phi_e : float
        flux offset
    """
    d = (gam-1)/(gam+1)
    A = np.pi*(phi + phi_e)/phi0
    B = Ejmax*np.abs(np.cos(A))
    C = np.sqrt(1+(d*np.tan(A))**2)
    return B*C

def f_01_phi(phi, Ejmax, gam, phi_e, Ec):
    """
    Ec here is fixed.
    input:
      phi : 1D numpy array
        external flux as a function of quanta
      Ejmax : float
        Total Ej in units of Hz
      d : float
        Junction asymmetry parameter
      phi_e : float
        flux offset
      Ec : float
        charging energy in units of Hz
    return:

    """
    f01 = np.sqrt(8*Ej_phi(phi, Ejmax, gam, phi_e)*Ec) - Ec
    return f01

def f_12_phi(phi, Ejmax, gam, phi_e, Ec):
    """
    Ec here is fixed.
    input:
      phi : 1D numpy array
        external flux as a function of quanta
      Ejmax : float
        Total Ej in units of Hz
      d : float
        Junction asymmetry parameter
      phi_e : float
        flux offset
      Ec : float
        charging energy in units of Hz
    return:
    """
    f12 = f_01_phi(phi, Ejmax, gam, phi_e, Ec) - Ec
    return f12

def Ej1_Ej2_d(Ejmax, gam):
    """
    Extract Ej1, Ej2 and d from the split transmon
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

def rb_error(p, d):
    """
    Errors per clifford gate and errors per gate from RB
    
    Parameters
    ----------
    p : list of floats
        1-p is the rate of depolarization.
        p[0] => list of nominal values
        p[1] => list of std_values
    d : float
        d=2^n => dimensionality of the hilbert space n=sum(|0>, |1>).

    Returns
    -------
    dictionary: 
        float of cliffords nominal errors
        float of standard deviation clifford errors
        float of nominal gate errors 
        float of std_dev clifford gate errors.
    """
    p_u = ufloat(nominal_value=p[0], std_dev=p[1])
    r_clifford_u = (1-p_u)/(1-(1/d))
    r_gate_u = r_clifford_u/1.875 #number of clifford over number of physical gates per clifford 1.875
    
    dict_error = {'r_clifford': [r_clifford_u.nominal_value, r_clifford_u.std_dev],
                  'r_gate': [r_gate_u.nominal_value, r_gate_u.std_dev]}
    return dict_error

def pb_error(u, d):
    """
    Purity benchmarking based on 1Q and 2Q-RB measurements.

    Parameters
    ----------
    u : list of floats
        unitarities (similar to p).
    d : float
        d=2^n dimensionality of hilbert space.

    Returns
    -------
    dictionary:
        float of purity cliffords nominal errors
        float of standard deviation purity clifford errors
        float of nominal purity gate errors 
        float of std_dev purity gate errors.
    """
    u_u = ufloat(nominal_value=u[0], std_dev=u[1])
    r_inc_clif_u = ((d-1)/d)*(1-sqrt(u_u))
    r_inc_gate_u = r_inc_clif_u/1.875 #number of clifford over number of physical gates per clifford 1.875
    
    dict_error = {'pr_clifford': [r_inc_clif_u.nominal_value, r_inc_clif_u.std_dev],
                  'pr_gate': [r_inc_gate_u.nominal_value, r_inc_gate_u.std_dev]}
    return dict_error

def irb_error(pc, pref, d, gate):
    """
    Errors per clifford gate and errors per gate from RB
    
    Parameters
    ----------
    pc : list of floats
        pc[0] => fitted irb values
        pc[1] => fitted irb stdev
    pref : list of floats
        1-p is the rate of depolarization.
        p[0] => nominal value
        p[1] => std_dev
    d : float
        d=2^n => dimensionality of the hilbert space n=sum(|0>, |1>).
    gate : string
        name of gate
    Returns
    -------
    dictionary: 
        float of cliffords nominal errors
        float of standard deviation clifford errors
        float of nominal gate errors 
        float of std_dev clifford gate errors.
    """
    p_u = ufloat(nominal_value=pref[0], std_dev=pref[1])
    p_c = ufloat(nominal_value=pc[0], std_dev=pc[1])
    r_clifford_u = (1-p_u)/(1-(1/d))
    r_gate_u = r_clifford_u/1.875 #number of clifford over number of physical gates per clifford 1.875
    ir_clifford_u = (d-1)*(1-p_c/p_u)/d
    ir_gate_u = ir_clifford_u/1.875
    
    dict_error = {'r_clifford': [r_clifford_u.nominal_value, r_clifford_u.std_dev],
                  'r_gate': [r_gate_u.nominal_value, r_gate_u.std_dev],
                  'ir_clifford': [ir_clifford_u.nominal_value, ir_clifford_u.std_dev],
                  'ir_gate': [ir_gate_u.nominal_value, ir_gate_u.std_dev],
                  'Gate of interest': gate}
    return dict_error

def rb_errors_unp(param_lst, p_list, d):
    """
    Errors per clifford gate and errors per gate
    Code could handle float or 1D numpy arrays
    
    # with parameters involved, it wiser to use np.array([])
    only issue is that by saving dictionaries, retrieval treats datasets as
    2D numpy array
    Parameters
    ----------
    param_list : list 
        param_list[0] = 'params'
        param_list[1] = 1D numpy arrays
    p_list : list of 1D numpy arrays
        1-p is the rate of depolarization.
        p[0] => 1D numpy array of nominal values with sequence to parameters
        p[1] => 1D numpy array of std_values with sequence to parameters
    d : float
        d=2^n => dimensionality of the hilbert space n=sum(|0>, |1>).

    Returns
    -------
    dictionary: 
        float of cliffords nominal errors
        float of standard deviation clifford errors
        float of nominal gate errors 
        float of std_dev clifford gate errors.
    """
    p_uarr = ufloat.uarray(p_list[0], p_list[1])
    r_c_uarr = (1-p_uarr)/(1-(1/d)) # number of error per cliffords
    r_g_uarr = r_c_uarr/1.875 #number of clifford over number of physical gates per clifford 1.875
    dict_rb_error = {'r_clifford_n': unumpy.nominal_values(r_c_uarr),
                     'r_clifford_std': unumpy.std_devs(r_c_uarr),
                     'r_gate_n': unumpy.nominal_values(r_g_uarr),
                     'r_gate_std': unumpy.std_devs(r_g_uarr),
                     param_lst[0]: np.array(param_lst[1])}
    # is it a good idea to convert 1D numpy array to list apart from fast plotting?
    return dict_rb_error

def pb_errors_unp(param_lst, u_list, d):
    """
    Purity benchmarking based on 1Q-RB measurements, handling arrays

    Parameters
    ----------
    u : list of 1D numpy arrays
        unitarities (similar to p).
    d : float
        d=2^n dimensionality of hilbert space.

    Returns
    -------
    dictionary:
        float of purity cliffords nominal errors
        float of standard deviation purity clifford errors
        float of nominal purity gate errors 
        float of std_dev purity gate errors.
    """
    u_uarr = ufloat.uarray(u_list[0], u_list[1])
    r_inc_c_uarr = ((d-1)/d)*(1-sqrt(u_uarr))
    r_inc_g_uarr = r_inc_c_uarr/1.875 #number of clifford over number of physical gates per clifford 1.875
    
    dict_pb_error = {'p_clifford_n': unumpy.nominal_values(r_inc_c_uarr),
                     'p_clifford_std': unumpy.std_devs(r_inc_c_uarr),
                     'p_gate_n': unumpy.nominal_values(r_inc_g_uarr),
                     'p_gate_std': unumpy.std_devs(r_inc_g_uarr),
                     param_lst[0]: np.array(param_lst[1])}
    return dict_pb_error

def irb_errors_unp(param_lst, p_fit, p_ref, d):
    """
    With interleaved benchmarking
    
    # with parameters involved, it wiser to use np.array([])
    only issue is that by saving dictionaries, retrieval treats datasets as
    2D numpy array
    Parameters
    ----------
    param_list : list 
        param_list[0] = 'params'
        param_list[1] = 1D numpy arrays
        param_list[2] = 'gate name'
    p_list : list of 1D numpy arrays
        1-p is the rate of depolarization.
        p[0] => 1D numpy array of nominal values with sequence to parameters
        p[1] => 1D numpy array of std_values with sequence to parameters
    d : float
        d=2^n => dimensionality of the hilbert space n=sum(|0>, |1>).

    Returns
    -------
    dictionary: 
        float of cliffords nominal errors
        float of standard deviation clifford errors
        float of nominal gate errors 
        float of std_dev clifford gate errors.
    """
    p_carr = ufloat.uarray(p_fit[0], p_fit[1])
    p_uarr = ufloat.uarray(p_ref[0], p_ref[1]) 
    r_c_uarr = (1-p_uarr)/(1-(1/d)) # number of error per cliffords in rb
    r_g_uarr = r_c_uarr/1.875 #number of clifford over number of physical gates per clifford 1.875
    ir_c_uarr = (d-1)*(1-p_carr/p_uarr)/d
    ir_g_uarr = ir_c_uarr/1.875
    param_string = param_lst[2]+'_'+ param_lst[0]
    
    dict_irb_errors = {'r_clifford_n': unumpy.nominal_values(r_c_uarr),
                       'r_clifford_std': unumpy.std_devs(r_c_uarr),
                       'r_gate_n': unumpy.nominal_values(r_g_uarr),
                       'r_gate_std': unumpy.std_devs(r_g_uarr),
                       'ir_clifford_n': unumpy.nominal_values(ir_c_uarr),
                       'ir_clifford_std': unumpy.std_devs(ir_c_uarr),
                       'ir_gate_n': unumpy.nominal_values(ir_g_uarr),
                       'ir_gate_std': unumpy.std_devs(ir_g_uarr),
                       param_string : np.array(param_lst[1])}
    # is it a good idea to convert 1D numpy array to list apart from fast plotting?
    return dict_irb_errors