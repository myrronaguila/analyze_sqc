# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:01:46 2024

lm_tdm stands for lmfit - time domain measurements. 
Highlighted fits include
1. T1 decay (Rabi and Exponential envelope)
2. T2 decay (Ramsey)
3. T2* decay (Spin-Echo)  

Toolbox for iterative time-domain measurements 
(will be coupled with other tools for complete iterative fitting packages
 for calibrating superconducting processors)

Key References:
    1. Matthew Neville intro to subclass: https://lmfit.github.io/lmfit-py/examples/example_complex_resonator_model.html#sphx-glr-examples-example-complex-resonator-model-py
    2. Instructive class https://mike.depalatis.net/blog/lmfit.html

Hopefully, itinerant fits such as T1 will be finished by monday night
at least not on an enbached settings. The idea is that I replicated the codes
for next week.

v1 => key definitions of key fits
v2 => class-based programming for lmfit for deployment with simulation models
v3 => with integration of fast-turn-around libraries

# adage. if lmfit has the package, use them as they are robust. If the model
is special (i.e. IQ clouds, then we make complex-resonator based fits)

In composing LMFIT time domain fitting functions, we follow the following format:
    1. for single-data fit (non-complicated function)
        lm_func(xdata, ydata, guess_dict=None, show=['N','N'], **kwargs)
        **kwargs => save=['N', filename], #save data for complicated functions 
                    input_info = ['par_name_1', val_1, etc]  
                    #input measure parameters that can be interleaved
        
        1) begin with built-in models or Model(user-defined models)
        2) Retrieve Guess Settings
        3) Set up Model Parametrs
        4) get results
        5) show fit reports if needed, we follow mike despandice to save text reports
        6) other outputs
        7) output best fit, dictionary of parameters
    2. for auto-fits (useful in analyzing 2D intensity maps)
        1) setup for loops with delay times
        2) put timing and benchmarks, and limit of the range of investigated peaks
        3) if data is run at vscode, graphical fits and report should be provided
        #no need for fit reports
        4) 
        5) give option to save dictionaries if fits are satisfactory
        7) output dictionary of extracted information, along with variable inputs

We hope the analysis toolbox here helps in scaling-up analysis beyond 1D of
code. Basic fits here correspond to one data-point in multiple-data-points.
These tools help to elucidate these parameters for appreciation.

Versions 
20250221 - added functionalites of lm_wave_c2 to consider a constrained 
sine-wave provided the magnitude is 1 

@author: Mai
"""
import time as time
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

from lm_utils import show_report
from lmfit.models import ConstantModel, SineModel, ExponentialModel
from uncertainties import ufloat, unumpy
from uncertainties.umath import sqrt
from lmfit import Model, Parameters

from lineshapes_qm import (exp_decay_cIQ, decay_cos_c, 
                           decay_sin_c, decay_wave_cIQ, gaussian_exp_cfit)
from lineshapes_qm import cosine

from scipy.signal import find_peaks

"""fundamental constants"""
kb = 1.380649E-23 #J K^-1
hbar = 1.054571817E-34 #J s

fit_method = {0: 'least_sq',
              1: 'least-squares',
              2: 'differential_evolution',
              3: 'nelder',
              4: 'bfgs',
              5: 'powell',
              6: 'ampgo',
              7: 'shgo',
              8: 'basinhopping',
              9: 'dual-annealing',
              10: 'trust-constr',
              11: 'emcee'}
"""--------------------------lmfit automated Time-Domain Models -----------"""

"""----------------Basic - Guess definitions--------------------------------"""
def guess_amp_decay(data, t, **kwargs):
    """
    # https://github.com/lmfit/lmfit-py/blob/master/lmfit/models.py
    # for I and Q signals
    
    Guess function for amplitude and decay based on lmfit. 
    Good guess function for getting exponential, but no good for getting
    amplitude.
    
    Limited uses for the following functions: 

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    try:
        sval, oval = np.polyfit(t, np.log(abs(data)+1.e-15), 1)
    except TypeError:
        sval, oval = 1., np.log(abs(max(data)+1.e-9))
    return np.exp(oval), -1.0/sval

def guess_decay_wave(data,t, wave='cos', show=['N','N'], **kwargs):
    """
    initial guess for decaying sine and cosine wave using lmfit inbuilt guess
    function. 
    
    While lmfit decaying sine wave is demonstrated to capture non-noisy signals in 
    its documentation, it fails to give a good initial guess for very noisy
    signals. Hence, we need to condition noisy signals to get a
    good guess of its decays. Hence, the output dictionary educates the 
    fitter about better bounds once a good decay constant, frequency shift 
    and amplitude is selected.
    
    Useful references: https://stackoverflow.com/questions/1713335/peak-finding-algorithm-for-python-scipy

    Parameters
    ----------
    data : 1D numpy array
        y-data or amplitude.
    t : 1D numpy array
        time.
    wave : string, optional
        Type of Wave. Restricted to sin and cos functions. The default is 'cos'.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    dict_guess : dictionary of guesses
        Guesses for decaying sinusoid wave.
    """
      
    """Use lmfit sine and cosine with offset to get offset, frequency and phase
    Found to be robust with fitting noisy sinusoids due to simplicity of the
    model"""
    
    fit_wave, dict_wave = lm_wave_c(xdata=t, ydata=data, show=show, wave=wave)
    #if wave='sin', par_list = [const_c, sin_amplitude, sin_frequency, sin_shift]
    #if wave='cos', par_list = [const_c, cos_amplitude, cos_frequency, cos_shift]
    
    pref0 = wave + '_'
    # get nominal value from fitted dictionary
    offs_val = dict_wave['const_c'][0]
    fd_val = dict_wave[pref0+'frequency'][0]
    phi_val = dict_wave[pref0+'shift'][0] % (2 * np.pi)
    
    """Use best fit wave to get the sample peaks location from raw_data"""
    amp = np.amax(fit_wave) - np.amin(fit_wave)
    idx_peaks, _ = find_peaks(x=fit_wave, prominence=amp/2)
    t_peaks = t[idx_peaks]
    y_peaks = data[idx_peaks] - dict_wave['const_c'][0]
    
    rep = 'N' # for debugging purpose only
    if rep == 'Y':
        plt.plot(t,fit_wave)
        plt.plot(t[idx_peaks], fit_wave[idx_peaks], 'x')
        plt.plot(t, data)
        plt.plot(t[idx_peaks], data[idx_peaks], 'x')
        plt.show()
    
    """Optional 1: Activate decay fit curve with saturation - best fit so far"""
    pref1 = 'exp_'
    exp_mod = ExponentialModel(prefix=pref1)
    pars_exp = exp_mod.guess(y_peaks, x=t_peaks)
    exp_res = exp_mod.fit(y_peaks, params=pars_exp, x=t_peaks)
    
    if rep == 'Y':
        #get report
        print(exp_res.fit_report(min_correl=0.10)) #T0=58.461 and exp_amp=2.5962
        # see if the noisy peaks can be fit with exponential
        plt.plot(t_peaks, y_peaks)
        plt.plot(t_peaks, exp_res.best_fit)
        plt.show()
    
    #results of exponential fit
    amp_val = exp_res.params[pref1+'amplitude'].value
    T0_val = exp_res.params[pref1+'decay'].value
    
    """Optional 2: Use in-built lmfit exponential with offset
    This approach is too noisy
    """
    # y_expc_fit, dict_e = lm_exp_decay_c(xdata=t_peaks, ydata=y_peaks, show=['N','N'])
    # amp_val = dict_e['exp_amplitude'][0]
    # T0_val = dict_e['exp_decay'][0]
    
    """Option 3: Use guess_decay and amplitude for answers - 
    cannot fit lmfit2"""
    # amp_val, T0_val = guess_amp_decay(data=y_peaks, t=t_peaks) #weirdly it leads to bad peaks
    # T0_val = np.abs(T0_val)
    # amp_val = np.abs(amp_val)

    """make dictionary of initial values"""
    dict_guess = {'offs': offs_val,
                  'T0': T0_val,
                  'fd': fd_val,
                  'amp': amp_val,
                  'phi': phi_val}
    return dict_guess

def guess_gauss_exp_c(data, t, T0, show=['N','N'], **kwargs):
    
    """Use Gaussian Guess Function to determine amplitude"""
    
    offs_val = np.average(data)
    amp_val = np.abs(np.amax(data) - np.amin(data))
    #amp_val = guess_amp_decay(data, t)[0]
    # try shifts in the range [0, 2*T0) and take the one with best residual
    pts_guess = 101 #improved but slow
    Tg_guesses = np.linspace(1.0, 2*T0, pts_guess, endpoint=False)
    #print(Tg_guesses)
    errors = [np.linalg.norm(gaussian_exp_cfit(t=t, Tg=Tg_guess, 
                                               Ioffs=offs_val, 
                                               Iamp=amp_val, 
                                               T0=T0) - data)
              for Tg_guess in Tg_guesses]
    # getting the best residual from linalg.norm
    
    Tg_val = Tg_guesses[np.argmin(errors)]
        
    dict_guess = {'Ioffs': offs_val,
                  'Tg': Tg_val,
                  'Iamp': amp_val}
    return dict_guess

"""----------------Part 0 - Classes of Basic Fits---------------------------"""
class CosineModel(lm.model.Model):
    r"""A model based on a sinusoidal lineshape.

    The model has three Parameters: `amplitude`, `frequency`, and `shift`.

    .. math::

        f(x; A, \phi, f) = A \cos (f x + \phi)

    where the parameter `amplitude` corresponds to :math:`A`, `frequency` to
    :math:`f`, and `shift` to :math:`\phi`. All are constrained to be
    non-negative, and `shift` additionally to be smaller than :math:`2\pi`.

    """
    
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(cosine, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        # same as np.cosine
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('frequency', min=0)
        self.set_param_hint('shift', min=-2*np.pi-1.e-5, max=2*np.pi+1.e-5)

    def guess(self, data, x, **kwargs):
        """Estimate initial model parameter values from the FFT of the data.
        - forked from lmfit.models
        """
        data = data - data.mean()
        # assume uniform spacing
        frequencies = np.fft.fftfreq(len(x), abs(x[-1] - x[0]) / (len(x) - 1))
        fft = abs(np.fft.fft(data))
        argmax = abs(fft).argmax()
        amplitude = 2.0 * fft[argmax] / len(fft)
        frequency = 2*np.pi * abs(frequencies[argmax])
        # try shifts in the range [0, 2*pi) and take the one with best residual
        pts_guess = 11 #default
        pts_guess = 101 #improved but slow
        shift_guesses = np.linspace(0, 2*np.pi, pts_guess, endpoint=False)
        errors = [np.linalg.norm(self.eval(x=x, amplitude=amplitude,
                                           frequency=frequency,
                                           shift=shift_guess) - data)
                  for shift_guess in shift_guesses]
        # getting the best residual from linalg.norm
        
        shift = shift_guesses[np.argmin(errors)]
        pars = self.make_params(amplitude=amplitude, frequency=frequency,
                                shift=shift)
        return lm.models.update_param_vals(pars, self.prefix, **kwargs)

class exp_decay_cIQ_Model(lm.model.Model):
    #__doc__= "T1_DecayModel" + lmfit.model.COMMON_INIT_DOC
    # getting the attributes from lmfit.model.Model => inheritance of Class
    # complex exponential decay constant, updating models to act like an in-built Model
    # used for composite functions like ConstantModel, etc in LMFIT
    
    # preserve this model
    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 *args, **kwargs):
        #pass in the defining equation so that the user doesn't have to later write the equation
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super().__init__(exp_decay_cIQ, *args, **kwargs)
        self.set_param_hint('T0', min=0) # enforce T0 is positive
        #self.vary_info = ['T0', 'Ioffs', 'Iamp', 'Qoffs', 'Qamp'] 
        # already in lmfit.model.Model as self._params
        
        # key issue with eval_components is that if the starting model is not
        # a composite function, then it will only release a built-in model
        
        #self._set_paramhints_prefix()
    
    def guess_c(self, data, t=None, **kwargs):
        """
        guess function under T0DecayModel, with control over bounds. 
        Physically reliable but quite unreliable at T0 longer than the range.
        
        (inspired from lmfit guess protocol for exponential decay. Problem
         is that it automatically modifies lmfit parameters. Okay. This part
         is special. For complex functions, we use LMFIT default. For 
         non-complex functions, we use my format)
        
        # remove in the final version
        
        Parameters
        ----------
        data : 1D numpy array of complex numbers
            y-axis (voltage or state probability).
        t : 1D numpy array, optional
            delay time between drive and readout pulses. The default is None.
        **kwargs : key-argument dictionary
            # must be omitted
            vary_bool = [True, True, True, True, True] list of booleans for 
            vary_info strings
            vary_info = ['T0', 'Ioffs', 'Iamp', 'Qoffs', 'Qamp']
        Returns
        -------
        lmfit param_vals.
        """
        
        verbose = kwargs.pop('verbose', None)
        # vary_info = ['T1', 'Ioffs', 'Iamp', 'Qoffs', 'Qamp']
        vary_info = self._param_names #inherited traits
        if 'vary_bool' in kwargs:
            vary_bool = kwargs['vary_bool']
        else:
            # switch to default
            vary_bool = [True, True, True, True, True] #default
        
        #add consideration for variations
        
        if t is None:
            return
        tmin = t.min()
        tmax = t.max()
        
        # for long T1 and offsets, we get the end-offset decays
        I_data = np.real(data) # get I data from real components
        Q_data = np.imag(data) # get Q data from comple components
        
        # get guess from I data
        Ioffs_guess = I_data[-1] #rather than data.mean, we get the saturation point
        Iamp_guess = I_data[0]-I_data[-1] #amplitude can be negative or positive
        Iamp_max = np.abs(np.amax(I_data) - np.amin(I_data))
        tau_I = guess_amp_decay(data=I_data-Ioffs_guess, t=t)[1] #amp has bad guess
                
        # get guess from Q data
        Qoffs_guess = Q_data[-1] #offset appreciated from end
        Qamp_guess = Q_data[0]-Q_data[-1] #amplitude can be negative or positive
        Qamp_max = np.abs(np.amax(Q_data) - np.amin(Q_data)) #assume positive values
        tau_Q = guess_amp_decay(data=Q_data-Qoffs_guess, t=t)[1] #amp has bad guess
        
        tau_guess = np.average(np.asarray([tau_I, tau_Q]))
        tau_max = 5*(tau_I + tau_Q) # max bound of T1
        
        if verbose:
            print(f"tmin={tmin}, tmax={tmax}, T1={tau_guess}")
            print(f"Iamp_guess={Iamp_guess}, Iamp_max={Iamp_max}, Ioffs_guess={Ioffs_guess}")
            print(f"Qamp={Qamp_guess}, Qamp_max={Qamp_max}, Qoffs_guess={Qoffs_guess}")
            print('\n')
        
        #make inital params from guess
        params = self.make_params(Ioffs=Ioffs_guess, Iamp=Iamp_guess, 
                                  Qoffs=Qoffs_guess, Qamp=Qamp_guess, 
                                  T1=tau_guess)
        
        #make bounds and constraints according to earlier inputs using set
        params[f'{self.prefix}'+vary_info[0]].set(min=tmin, max=tau_max, vary=vary_bool[0])
        params[f'{self.prefix}'+vary_info[1]].set(vary=vary_bool[1]) #I_offs
        params[f'{self.prefix}'+vary_info[2]].set(max=Iamp_max, vary=vary_bool[2])
        params[f'{self.prefix}'+vary_info[3]].set(vary=vary_bool[3]) #Q_offs
        params[f'{self.prefix}'+vary_info[4]].set(max=Qamp_max, vary=vary_bool[4])
        return lm.models.update_param_vals(params, self.prefix, **kwargs)
    
    def guess(self, data, t=None, **kwargs):
        """
        Copy of lmfit exponential + constant guess function but applied to IQ
        data

        Parameters
        ----------
        data : 1D numpy array of complex numbers
            y-axis (voltage or state probability).
        t : 1D numpy array, optional
            delay time between drive and readout pulses. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        params for lmfit

        """
        # use guess done on constant functions
        
        # retrieve I and Q from data
        I_data = np.real(data) 
        Q_data = np.imag(data) 
        
        Ioffs_g = I_data.mean()
        Iamp_g, tau_I = guess_amp_decay(I_data, t, **kwargs)
        
        Qoffs_g = Q_data.mean() 
        Qamp_g, tau_Q = guess_amp_decay(Q_data, t, **kwargs)
        
        tau_g = np.abs((tau_I + tau_Q)/2) #get average time-constant must be positive
                
        # make inital params from guess - no bounds
        params = self.make_params(Ioffs=Ioffs_g, Iamp=Iamp_g, 
                                  Qoffs=Qoffs_g, Qamp=Qamp_g, 
                                  T0=tau_g)
                
        # we set bounds when there is an issue
        return lm.models.update_param_vals(params, self.prefix, **kwargs)

class decaying_wave_cIQ_Model(lm.model.Model):
    #__doc__= "T1_DecayModel" + lmfit.model.COMMON_INIT_DOC
    # getting the attributes from lmfit.model.Model => inheritance of Class
    # complex exponential decay constant, updating models to act like an in-built Model
    # used for 
    def __init__(self, independent_vars=['t'], prefix='', nan_policy='raise',
                 *args, **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})        
        #pass in the defining equation so that the user doesn't have to later write the equation
        super().__init__(decay_wave_cIQ, *args, **kwargs)
        self.set_param_hint('T0', min=0) # enforce T1 is positive
        #self.vary_info = ['T0', 'freq', 'phi', 'Ioffs', 'Iamp', 'Qoffs', 'Qamp'] 
        # already in lmfit.model.Model as self._params
    
    def _set_paramhints_prefix(self):
        # added today
        # same as np.cosine, initial parameters
        self.set_param_hint('phi', min=-2*np.pi-1.e-5, max=2*np.pi+1.e-5)
        self.set_param_hint('Iamp', min=0)
        self.set_param_hint('Qamp', min=0)
        # self.set_param_hint('fd', min=0)
        # not constraining fd to be negative as this can be a detuning.
        
    def guess(self, data, t=None, **kwargs):
        """
        Copy of lmfit exponential + constant guess function but applied to IQ
        data

        Parameters
        ----------
        data : 1D numpy array of complex numbers
            y-axis (voltage or state probability).
        t : 1D numpy array, optional
            delay time between drive and readout pulses. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        params for lmfit

        """
        
        # retrieve I and Q from data
        I_data = np.real(data) 
        Q_data = np.imag(data) 
        
        #retrieve guess from home-made functions
        dict_Ig = guess_decay_wave(I_data,t, wave='cos', show=['N','N'])
        dict_Qg = guess_decay_wave(Q_data,t, wave='sin', show=['N','N'])
        
        """
        dict_guess = {'offs': offs_val,
                      'T0': T0_val,
                      'fd': fd_val,
                      'amp': amp_val,
                      'phi': phi_val}
        """
        
        # fd_g = (dict_Ig['fd']+dict_Qg['fd'])/2
        # phi_g = (dict_Ig['phi']+dict_Qg['phi'])/2
        fd_g = dict_Ig['fd']
        phi_g = dict_Ig['phi']
        tau_g = np.abs((dict_Ig['T0'] + dict_Qg['T0'])/2) #get average time-constant must be positive
        Ioffs_g = dict_Ig['offs']
        Iamp_g = dict_Ig['amp']
        Qoffs_g = dict_Qg['offs']
        Qamp_g= dict_Qg['amp']
        # make inital params from guess - no bounds
        params = self.make_params(Ioffs=Ioffs_g, Iamp=Iamp_g, 
                                  Qoffs=Qoffs_g, Qamp=Qamp_g, 
                                  T0=tau_g, fd=fd_g, phi=phi_g)
                
        # we set bounds when there is an issue
        """
        # from lm_decaying_wave_c
        pars.add_many((par_list[0], dict_g['T0'], True, 1.0, 1E6),
                      (par_list[1], dict_g['fd'], True),
                      (par_list[2], dict_g['phi'], True),
                      (par_list[3], dict_g['amp'], True, 0.0, np.inf),
                      (par_list[4], dict_g['offs'], True))
        """
        
        params[f'{self.prefix}'+'T0'].set(min=1.0, max=1E6, vary=True)
        params[f'{self.prefix}'+'Iamp'].set(min=0.0, max=np.inf, vary=True)
        params[f'{self.prefix}'+'Qamp'].set(min=0.0, max=np.inf, vary=True)
        
        #added 20240329
        params[f'{self.prefix}'+'phi'].set(min=-2*np.pi-1.e-5, max=2*np.pi+1.e-5, vary=True)
        # params[f'{self.prefix}'+'fd'].set(min=0, max=np.inf, vary=True)
        
        return lm.models.update_param_vals(params, self.prefix, **kwargs)

"""---------------------------Part 1 - Relaxation Fits----------------------"""
"""------------------Exponential Decay for 1 channel, I and State-----------"""
def lm_exp_decay_c(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    Exponential fit with baseline using lmfit built-in functions
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        real amplitude.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const_c)
            par_vary[1] = True or False (boolean, on exp_decay)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list - becomes useful for multivariate data
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    """Set-up model using the build-in functions"""
    const_mod = ConstantModel(prefix='const_') 
    exp_mod = ExponentialModel(prefix='exp_') 
        
    """Put guess parameters using composite functions"""
    par_list = ['const_c', 'exp_decay', 'exp_amplitude']
    
    #perform if-else for fit
    if ('guess_dict' in kwargs) and ('par_vary' in kwargs):
        pars = Parameters()
        dict_g = kwargs['guess_dict'] # dictionary
        par_vary = kwargs['par_vary'] # list of lists
        pars.add_many((par_list[0], dict_g['const_c'][0], par_vary[0], 
                       dict_g['const_c'][1], dict_g['const_c'][2]),
                      (par_list[1], dict_g['exp_decay'][0], par_vary[1], 
                       dict_g['exp_decay'][1], dict_g['exp_decay'][2]),
                      (par_list[2], dict_g['exp_amplitude'][0], par_vary[2], 
                       dict_g['exp_amplitude'][1], dict_g['exp_amplitude'][2]))
    else:
        pars = const_mod.guess(ydata, x=xdata) # takes the average - works great
        pars += exp_mod.guess(ydata, x=xdata) #exp function in lmfit is more robust
    # future implementation: rename functions 
    
    """Const_mod and Exp_Mod guess are way off but the resulting fits are quite
    robust even if the values are negative"""
    
    """Add Models and Evaluate"""
    mod = const_mod + exp_mod
    
    # to save lines reports will be under lm_utils
    out = mod.fit(ydata, pars, x=xdata)
    
    #show_report(result, xdata, ydata, show=['N', 'N'], **kwargs)
    show_report(result=out, xdata=xdata, ydata=ydata, show=show) #works as intended
    
    """Report Results - quite verbose, need a function to reduce words - 
    report to be delegated in lm-utils. Plotting uses data-qm method - this 
    will be my coding library - like that of the resonator-tools library 
    package - removed in the next version as the show_report has been 
    successful"""
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                         out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    
    return out.best_fit, out_dict_val_stderr

"""--------------Exponential Decay for I and Q Data-------------------------"""
def lm_const(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    Cosine / Sine fit with baseline using lmfit built-in functions.
    Useful for guess initial parameters
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        complex amplitude.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    
    """Declare Model"""
    pref1 = 'const_' 
    const_mod = ConstantModel(prefix=pref1)
    par_list = [pref1+'c']
    # print(par_list)
    
    pars = const_mod.guess(data=ydata,x=xdata)
    
    # declare model
    mod = const_mod
    
    #fit model
    out = mod.fit(ydata, params=pars, x=xdata)
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    return out.best_fit, out_dict_val_stderr

def lm_wave_c(xdata, ydata, show=['Y','Y'], wave='cos', **kwargs):
    """
    Cosine / Sine fit with baseline using lmfit built-in functions.
    Useful for guess initial parameters
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        complex amplitude.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    np.random.seed(42)
    
    
    """Declare Model"""
    par_list0 = ['amplitude', 'shift', 'frequency', 'c']
    if wave == 'cos':
        pref0 = wave +'_'
        wave_mod = CosineModel(prefix=pref0)
    else:
        pref0 = 'sin_'
        wave_mod = SineModel(prefix=pref0)
    
    pref1 = 'const_' 
    const_mod = ConstantModel(prefix=pref1)
    
    """ Add guess functionality to constrain parameters """
    
    par_var = [True, True, True, True, False]
    weights = np.ones(len(ydata))
    method=fit_method[0] #Levenberg-marquardt
    axs_lbl_fit = ['x-data', 'y-data']
    dict_g = {}
    
    # Check boolean for params
    # if 'bool_params' in kwargs:
    #     par_var = kwargs['bool_params']
    # else:
    #     par_var = [True, True, True, True, False] # default unique to the Model
    #     # to generalize, put things in default 
    
    # # second version for guess; nested if-else statement for better guess
    # if 'guess' in kwargs:
    #     guess = kwargs['guess']
    #     if 'bool_params' in kwargs:
    #         # modifies guess depending on action of vary in lmfit
    #         for i in range(len(par_var)):
    #             if par_var[i] == False:
    #                 # manually-inputting guess function depending on par var
    #                 dict_g[par[i]] = guess[i]
    #             else:
    #                 dict_g[par[i]] = guess_f01_volts_to_hz(xdata,ydata,guess[4])[par[i]] # better at guessing parameters, works with gpg
    #     else:
    #         """
    #         guess based on intuition or on pre-fitted data, usually originating
    #         from published results, for testing reproducibility of model.
    #         """
    #         dict_g = {par[0]: guess[0],
    #                   par[1]: guess[1],
    #                   par[2]: guess[2],
    #                   par[3]: guess[3],
    #                   par[4]: guess[4]}
    # else:
    #     dict_g = guess_f01_volts_to_hz(xdata, ydata) # better at guessing parameters, works with gpg
    
    # # print(dict_g)
    
    # p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    par_list = [pref0 + par_list0[i] if i < 3 else pref1 + par_list0[i] for
                i in range(len(par_list0))]
    # print(par_list)
    
    pars = const_mod.guess(data=ydata,x=xdata)
    pars += wave_mod.guess(data=ydata,x=xdata)
    
    # declare model
    mod = wave_mod + const_mod
    
    #fit model
    out = mod.fit(ydata, params=pars, x=xdata)
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    time.sleep(0.1) # delay for autofit functions
    
    return out.best_fit, out_dict_val_stderr

def lm_wave_c2(xdata, ydata, show=['Y','Y'], wave='cos', **kwargs):
    """
    Cosine / Sine fit with baseline using lmfit built-in functions.
    Useful for guess initial parameters
    
    Note: constrained amplitude and offset. amplitude = 1 - offset
    - 20250303 -> turned out the use case is not useful in determining
    the junction asymmetry
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        complex amplitude.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    
    """Declare Model"""
    par_list0 = ['amplitude', 'shift', 'frequency', 'c']
    if wave == 'cos':
        pref0 = wave +'_'
        wave_mod = CosineModel(prefix=pref0)
    else:
        pref0 = 'sin_'
        wave_mod = SineModel(prefix=pref0)
    
    pref1 = 'const_' 
    const_mod = ConstantModel(prefix=pref1)
    par_list = [pref0 + par_list0[i] if i < 3 else pref1 + par_list0[i] for
                i in range(len(par_list0))]
    # print(par_list)
    
    # declare model
    mod = wave_mod + const_mod
    
    pars = const_mod.guess(data=ydata,x=xdata) # issue with pars
    pars += wave_mod.guess(data=ydata,x=xdata) # issue with pars
    
    # pars =  wave_mod.guess(data=ydata,x=xdata)
    # pars.update(const_mod.guess(data=ydata,x=xdata))
    
    # pars["const_c"].set(value=0.5, min=0, max=1)  # Ensuring 0 <= offset <= 1
    
    pars["const_c"].set(vary=True, min=0.5)  # Enforces const_c > 0.5, gpt_4 mod.
    
    if wave == 'cos':
        # pars.add("constraint", expr="cos_amplitude + const_c - 1") #gpt_1
        # pars.add("cos_amplitude", expr="1 - const_c") #gpt_2
        pars["cos_amplitude"].set(expr="1 - const_c") #gpt_3, worked but const_amp < const_c    
    else:
        # for sine functions with added constraints
        # pars.add("constraint", expr="sin_amplitude + const_c - 1") #gpt_1
        # pars.add("sin_amplitude", expr="1 - const_c") #gpt_2
        pars["sin_amplitude"].set(expr="1 - const_c") #gpt_3
    
    #fit model
    out = mod.fit(ydata, params=pars, x=xdata)
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    return out.best_fit, out_dict_val_stderr

def lm_exp_decay_cIQ(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    Exponential fit with baseline using lmfit built-in functions
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        complex amplitude.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    
    opt = True
    if opt==True:
        """Option 1: using home-made complex functions 
        Set-up model using the build-in functions"""
        #mod = exp_decay_cIQ_Model(prefix='m_) #prefix works
        mod = exp_decay_cIQ_Model()
        """Guess parameters"""
        par_list = ['T0', 'Ioffs', 'Iamp', 'Qoffs', 'Qamp']
        pars = mod.guess(ydata, t=xdata, verbose=True)
        out = mod.fit(ydata, params=pars, t=xdata)
    else:
        """Option 2: use in-built composite functions - not working. Remove in final version"""
        ci_mod = ConstantModel(prefix='ci_') 
        expi_mod = ExponentialModel(prefix='expi_')
        cq_mod = ConstantModel(prefix='cq_') 
        expq_mod = ExponentialModel(prefix='expq_')
        
        pars_i = ci_mod.guess(np.real(ydata),x=xdata)
        pars_i += expi_mod.guess(np.real(ydata), x=xdata)
        
        pars_q = cq_mod.guess(np.imag(ydata),x=xdata)
        pars_q += expq_mod.guess(np.imag(ydata), x=xdata)
        
        pars = pars_i + pars_q
        
        #Trial 1: result => TypeError: unsupported operand type(s) for *: 'complex' and 'CompositeModel'
        #mod = (ci_mod + expi_mod) + 1j*(cq_mod + expq_mod)
        #Trial 2: result
        mod = ci_mod + expi_mod + 1j*(cq_mod +expq_mod)
        out = mod.fit(ydata, pars, x=xdata)
        """Composite functions could not handled complex valued models.
        Go with opt=True. we will remove these"""
        
    """Plot result"""
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    
    return out.best_fit, out_dict_val_stderr

"""---Part 2: - Decaying I and Q functions - Rabi and Ramsey fringe---------"""
def lm_decaying_wave_c(xdata, ydata, show=['Y','Y'], wave='cos', **kwargs):
    """
    Exponential fit with cos/sin wave and baseline using home-made functions
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        real amplitudes.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    
    """Option 1: using home-made complex functions: Set-up model using the lmfit 
    functions. 
    
    Turned out to be not possible due to complicated setting so we use in-built
    lmfit functions to get good guess of our decaying cosine func. In the
    end, there is no eval components for this method. If we will do 
    decomposition, we have to use model.evals"""
    
    """Declare Model"""
    if wave == 'cos':
        mod = Model(decay_cos_c)
        par_list = ['T0', 'fd', 'phi_I', 'Iamp', 'Ioffs']
    else:
        wave = 'sin'
        mod = Model(decay_sin_c)
        par_list = ['T0', 'fd', 'phi_Q', 'Qamp', 'Qoffs']
        
    """Guess parameters based on dictionary"""
    dict_g = guess_decay_wave(data=ydata,t=xdata, wave=wave, **kwargs)
    
    """
    # Note from previous efforts
    pars = ['T0', 'Iamp', 'Ioffs', 'Qamp', 'Qoffs']
    params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    # time in nanosecond
    params.add_many((pars[0], p[0], True, 1.0, 1.0E6),
                    (pars[1], p[1], True, -np.inf, np.amax(dataRe) - np.amin(dataRe)),
                    (pars[2], p[2], True, -np.inf, np.amax(dataRe)),
                    (pars[3], p[3], True, -np.inf, np.amax(dataIm) - np.amin(dataIm)),
                    (pars[4], p[4], True, -np.inf, np.amax(dataIm)))
    """
    pars = Parameters()
    pars.add_many((par_list[0], dict_g['T0'], True, 1.0, 1E6),
                  (par_list[1], dict_g['fd'], True),
                  (par_list[2], dict_g['phi'], True, 0.0, 2*np.pi),
                  (par_list[3], dict_g['amp'], True, 0.0, np.inf),
                  (par_list[4], dict_g['offs'], True))
    #print(pars)
    """Output fit"""
    out = mod.fit(ydata, params=pars, t=xdata)
    """Plot result"""
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    return out.best_fit, out_dict_val_stderr

def lm_decaying_wave_cIQ(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    Exponential fit with cos/sin wave and baseline using home-made functions
    now with complex inputs
    
    xdata : 1D npy array
        time.
    ydata : 1D np array
        real amplitudes.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess_dict : dictionary of values 
            if guess_dict=None, we use lmfit well-defined values
        par_vary : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const)
            par_vary[1] = True or False (boolean, on offset)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        save : list of strings
            example: save=['N', 'file']
            save[0] => save or not
            save[1] => 'file-name of file'
        time : string 
            time='Y' or 'N' => display time or not (no need due to speed)
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
            length can be as long as possible
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_values:  dictionary of results
        best fit parameters
    """
    mod = decaying_wave_cIQ_Model()
    pars = mod.guess(data=ydata, t=xdata)
    par_list = ['T0', 'fd', 'phi', 'Iamp', 'Ioffs', 'Qamp', 'Qoffs']
        
    # print(pars)
    """Output fit"""
    out = mod.fit(ydata, params=pars, t=xdata)
    """Plot result"""
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    return out.best_fit, out_dict_val_stderr
    
def lm_gauss_exp_cfit(xdata, ydata, T0, show=['Y','Y'], **kwargs):
    """
    Gaussian Decay due to spin-echo
    
    In LMFit, Composite Function is ConstantModel() + GaussianModel() + ExponentialModel()

    Parameters
    ----------
    t : 1D numpy array
        time interval between two pulses.
    Tg : float
        fitting gaussian time errors.
    T0 : float
        Acquired T0.
    Ioffs : float
        Offset voltage.
    Iamp : float
        Measured Amplitude.

    Returns
    -------
    None.
    """
    
    """Declare Model"""
    mod = Model(gaussian_exp_cfit)
    par_list = ['Tg', 'Ioffs', 'Iamp', 'T0']
    
    """Guess parameters based on dictionary"""
    dict_g = guess_gauss_exp_c(data=ydata,t=xdata, T0=T0, **kwargs)
        
    """
    # Note from previous efforts
    pars = ['T0', 'Iamp', 'Ioffs', 'Qamp', 'Qoffs']
    params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    # time in nanosecond
    params.add_many((pars[0], p[0], True, 1.0, 1.0E6),
                    (pars[1], p[1], True, -np.inf, np.amax(dataRe) - np.amin(dataRe)),
                    (pars[2], p[2], True, -np.inf, np.amax(dataRe)),
                    (pars[3], p[3], True, -np.inf, np.amax(dataIm) - np.amin(dataIm)),
                    (pars[4], p[4], True, -np.inf, np.amax(dataIm)))
    """
    
    """Create Parameters"""
    pars = Parameters()
    pars.add_many((par_list[0], dict_g['Tg'], True, 1.0, 1E6),
                  (par_list[1], dict_g['Ioffs'], True),
                  (par_list[2], dict_g['Iamp'], True, 0.0, np.inf),
                  (par_list[3], T0, False))
    # print(par_list)
    # print(pars)
    
    # print(pars)
    """Output fit"""
    out = mod.fit(ydata, params=pars, t=xdata)
    """Plot result"""
    show_report(result=out, xdata=xdata, ydata=ydata, show=show, **kwargs)
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                          out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    
    return out.best_fit, out_dict_val_stderr

"""--------------------Notes for RB and 2RB fits, use pyQuaill--------------
Better we learn tricks first from readout fits.
"""


"""---------------------1Q sequence fidelities------------------------------"""
def rb_1Q_seq_fid(m, A, p, B):
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
    return rb_1Q_seq_fid(m=m-1, A=Ap, p=u, B=Bp)

def rb_error(p, d):
    """
    Errors per clifford gate and errors per gate from RB, 1 value
    
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
    r_gate_u = r_clifford_u/1.875 
    #number of clifford over number of physical gates per clifford 1.875
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
    Errors per clifford gate and errors per gate from RB, 1 value
    
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