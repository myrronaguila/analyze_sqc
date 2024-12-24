# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:25:48 2024

lmfit of deterministic benchmarking

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from data_plots_qm import cm_to_inch
from scipy.stats import linregress
from lmfit import Model, Parameters
from lm_utils import show_report
from uncertainties import ufloat
from lineshapes_det_bench import dB_errors, dB_fidelity
from lineshapes_det_bench import get_db_errors

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

"""--------------------1. Guess Functions for Modified Data-----------------"""
def guess_amp_decay(data, t):
    """
    # https://github.com/lmfit/lmfit-py/blob/master/lmfit/models.py
    # for I and Q signals
    
    Guess function for amplitude and decay in a decay curve based on lmfit. 
    Good guess function for getting exponential, but no good for getting
    amplitude in a sine / cosine wave.
    
    Limited uses for the following functions: 
        
    # lmfit guess function is bad.

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
    amp_decay: float
        decay amplitude from decay fit.
    T_decay : float
        decay time.

    """
    try:
        sval, oval = np.polyfit(t, np.log(abs(data)+1.e-15), 1)
    except TypeError:
        sval, oval = 1., np.log(abs(max(data)+1.e-9))
    return np.exp(oval), -1.0/sval

def guess_amp_decay_2(data, t):
    y_adj = data - data-np.min(data)
    
    # guess the amplitude: maximum of adjusted data
    A_guess = np.amax(y_adj)
    
    # guess for the decay time.
    threshold = A_guess/np.exp(1)
    idx_above_threshold = np.where(y_adj >= threshold)
    if len(idx_above_threshold) > 1:
        t1, t2 = t[idx_above_threshold[0]], t[idx_above_threshold[-1]]
        tau_guess = (t2 - t1) / np.log(y_adj[idx_above_threshold[0]] / y_adj[idx_above_threshold[-1]])
    else: 
        tau_guess = (t[-1] - t[0]) / 2  # Fallback if insufficient data
    return A_guess, tau_guess

def guess_dB_errors(t, ydata):
    """
    Extract variables from a. Best way to determine a is to use a decaying
    cosine wave without phase and subtract A and B. 
    
    Need guess function to be much more reader friendly
    
      Parameters
      ----------
      n : 1D numpy array of photon number
      ydata : 1D numpy array
      Qi,inv
    
    Returns
    -------
    """
    # determine f, constraint = [0, np.inf]
    
    # get average wave if it is sinusoid
    data = ydata - ydata.mean()
    
    # assume uniform spacing
    frequencies = np.fft.fftfreq(len(t), abs(t[-1] - t[0]) / (len(t) - 1)) #chatgpt
    fft = abs(np.fft.fft(data))
    argmax = abs(fft).argmax()
    f_guess = abs(frequencies[argmax])/2
    
    # determine exponential amplitue and decay constant
    # amp_exp, T_guess = guess_amp_decay(ydata, t=t), inaccurate, old from scipy
    amp_exp, T_guess = guess_amp_decay_2(ydata, t=t)
    
    # determine a, constraint = [-1, 1], measure amp and base
    if f_guess > 0:
        # A = 2.0 * fft[argmax] / len(fft)
        A = np.amax(np.abs(data)) #better solution
        B = ydata.mean()
    else: 
        A = amp_exp
        B = ydata.min()
    #A = 0.5*(1-a), B = 0.5*(1+a) B-A = a
    a_guess = B - A
    
    # determine T_d = [0 ns, 1 ms]
                
    dict_g = {'f_e': f_guess,
              'T_d': T_guess,
              'a': a_guess}
    return dict_g

def guess_dB_infid(n, ydata, num_qubits=1):
    """
    
    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.

    Returns
    -------
    dict_g. dictionary
        dictionary of initial guesses
    """
    # guess B errors
    B_guess = np.amin(ydata)
    
    # guess amplitude
    y_adj = ydata - B_guess
    
    # do logarithmic conversion, ensure no negative values
    valid_indices = y_adj > 0
    log_y = np.log10(y_adj[valid_indices])
    n_arr = n[valid_indices]
    # obtain slope via the form log(y_adj) = log10(A) + n*log10(p), where slope is log10(p)   
    log_p, log_A, _, _, _ = linregress(x=n_arr, y=log_y)
    
    # determine amplitude
    A_guess = 10**log_A
    
    # determine decay parameter
    p_guess = 10**log_p
    
    # determine infidelity
    d = 2**num_qubits
    r_guess = (d-1)*(1-p_guess)/d
    dict_g = {'A': A_guess,
              'B': B_guess,
              'r_clif': r_guess,
              'num_qubits': num_qubits}
    return dict_g
    
"""-------------------------lmfit.model version-----------------------------"""
def lm_dB_errors(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    lmfit.Model version of empirical dB benchmarking

    Parameters
    ----------
    xdata : 1D numpy array
        Evolution time in nanoseconds.
    ydata : 1D numpy array
        Fidelity.
    show : list of strings, optional
        show[0]='Y' => show fit report in text
        show[1]='Y' => show best fit plot. The default is ['Y','Y'].
    **kwargs : dictionary
        List of optional modes.
        bool_params : list of booleans
            bool_params[0] = True or False (for f_e)
            bool_params[1] = True or False (for T_d)
            bool_params[2] = True or False (for a)
            bool_params[3] = True or False
        guess : list of float
            guess[0] => f_e (frequency related to bloch vector misalignment)
            guess[1] => T_d (time scale in nanosecond)
            guess[2] => a (related to finite temperature comparable with single-shot measurement)
        weights : 1D numpy array
            examples: np.power(ydata,-1) #bias
            np.power(ydata, -0.5) # bias
            np.ones(len(ydata)) => default
        method : string
            Fitting algorithm (i.e. 'least-square', 'nelder-mead', 'differential-evolution')
        fit_res_pts : float
            Number of fitting points different from xdata
    Returns
    -------
    lst_best_fit : 1D numpy array
        best-fit of curve in y-data
    out_dict_val_stderr : dictionary
        {fit_param_name_1 : nom_best_fit, err_best_fit,
         ...}
    """
    np.random.seed(42)
    
    # set model and parameters
    mod = Model(dB_errors)
    par = ['f_e', 'T_d', 'a'] #name of fitting variables
    
    # set parameter in guess if guess is inputted
    """More efficient to combine 'guess' and 'bool_params' in kwargs"""
    
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, True]
    
    dict_g = {}
    # first version for guess; only f_e and a are modifiable by input
    # if 'guess' in kwargs:
    #     # useful for predicting the time constant without any error
    #     guess = kwargs['guess'] #expressed in dictionary
    #     dict_g = {'f_e': guess[0],
    #               'T_d': guess_dB_errors(xdata, ydata)['T_d'],
    #               'a': guess[2]}
    # else: 
    #     dict_g = guess_dB_errors(xdata, ydata)
        
    # second version for guess; even T_d is modifiable for gate asymmetry
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_dB_errors(xdata,ydata)[par[i]]
        else:
            # guess based on intuition but not impeding variations of guess
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2]}
    else:
        dict_g = guess_dB_errors(xdata, ydata)
    
    # initialize guess
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    params.add_many((par[0], p[0], par_var[0], 0.0, 5/(np.abs(xdata[1] - xdata[0]))),
                    (par[1], p[1], par_var[1], 1.0, 1E6), 
                    (par[2], p[2], par_var[2], -1.1, 1.1))
    
    # fe = 0 to point resolution
    # Td = 1 to 1E6 ns
    # a = -1 to 1
        
    if 'weights' in kwargs:
        weights = kwargs['weights'] #weights, 1D numpy array
    else:
        weights = None
    
    if 'method' in kwargs:
        method = kwargs['method']
    else: 
        # method=fit_method[2] #Differential Evolution
        method=fit_method[0] #levenberg-marquardt
    
    out = mod.fit(ydata, params, tn=xdata, scale_covar=True, method=method, 
                  weights=weights)
  
    # add label to plot
    axs_lbl_fit = [r'Evolution time $t$ (ns)', r'Fidelity']
    
    # best fit must be outputed with better evaluation, later
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.geomspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        #print(out.params)
        out_eval = out.eval(tn=xdata_eval)
        
        # comp evals added for deconvoluted plots in case of compounded models
        comps_eval = out.eval_components(tn=xdata_eval)
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    fit_eval = [xdata_eval, out_eval, comps_eval], 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata_eval, out_eval]
    else: 
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata, out.best_fit]
    """Method - iterable dictionaries {} more efficient. Names are keys and the dictionary
    contains a list. [0] refers to best fit whereas [1] would be the standard error"""
    out_dict_val_stderr = {par[i]: [out.params[par[i]].value, out.params[par[i]].stderr]
                           for i in range(len(par))}
    
    time.sleep(0.1)
    return lst_best_fit, out_dict_val_stderr

# - RB-like and SPAM characterization
def lm_dB_infid(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    lm.model version of SPAM fidelity and randomized clifford errors

    Parameters
    ----------
    xdata : 1D numpy array
        Evolution time in nanoseconds.
    ydata : 1D numpy array
        Fidelity.
    show : list of strings, optional
        show[0]='Y' => show fit report in text
        show[1]='Y' => show best fit plot. The default is ['Y','Y'].
    **kwargs : dictionary
        List of optional modes.
        bool_params : list of booleans
            bool_params[0] = True or False (for A)
            bool_params[1] = True or False (for B)
            bool_params[2] = True or False (for r_clif => clifford error)
            bool_params[3] = True or False
        guess : list of float
            guess[0] => A (Initial State Probability)
            guess[1] => B (Final state probability)
            guess[2] => r_clif (clifford error)
            guess[3] => number of qubits (constant)
        weights : 1D numpy array
            examples: np.power(ydata,-1) #bias
            np.power(ydata, -0.5) # bias
            np.ones(len(ydata)) => default
        method : string
            Fitting algorithm (i.e. 'least-square', 'nelder-mead', 'differential-evolution')
        fit_res_pts : float
            Number of fitting points different from xdata
    Returns
    -------
    lst_best_fit : 1D numpy array
        best-fit of curve in y-data
    out_dict_val_stderr : dictionary
        {fit_param_name_1 : nom_best_fit, err_best_fit,
         ...}
    """
    np.random.seed(42)
    
    # set model and parameters
    mod = Model(dB_fidelity)
    par = ['A', 'B', 'r_clif', 'num_qubits'] #name of fitting variables
    
    # set parameter in guess if guess is inputted
    """More efficient to combine 'guess' and 'bool_params' in kwargs"""
    
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, True, False]
    
    dict_g = {}
    
    # second version for guess; even T_d is modifiable for gate asymmetry
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_dB_infid(xdata,ydata)[par[i]]
        else:
            # guess based on intuition but not impeding variations of guess
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2],
                      par[3]: guess[3]}
    else:
        dict_g = guess_dB_infid(xdata, ydata)
    
    # initialize guess
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    params.add_many((par[0], p[0], par_var[0], 0.0, 1.0),
                    (par[1], p[1], par_var[1], 0.0, 1.0), 
                    (par[2], p[2], par_var[2], 1E-8, 1.0),
                    (par[3], p[3], False))
    params.add('spam_F', expr='A+B')
    
    # A = 0 to 1.0
    # B = 0 to 1.0
    # r_clif = 1E-6 to 1.0
    # num_qubits=1
    
    # add parameters with constraints: SPAM
    # mod.set_param_hint(name='spam_err',expr='1-A-B')
    
    if 'weights' in kwargs:
        weights = kwargs['weights'] #weights, 1D numpy array
    else:
        weights = np.ones(len(ydata))
    
    if 'method' in kwargs:
        method = kwargs['method']
    else: 
        method=fit_method[0] #Levenberg-marquardt
        # method=fit_method[2] #differential errors => gives runtime errors
    
    out = mod.fit(ydata, params, n=xdata, scale_covar=True, method=method, 
                  weights=weights)
  
    # add label to plot
    axs_lbl_fit = [r'Number of cliffords (n)', r'Fidelity']
    
    # best fit must be outputed with better evaluation, later
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.geomspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        #print(out.params)
        out_eval = out.eval(n=xdata_eval)
        
        # comp evals added for deconvoluted plots in case of compounded models
        comps_eval = out.eval_components(n=xdata_eval)
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    fit_eval = [xdata_eval, out_eval, comps_eval], 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata_eval, out_eval]
    else: 
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata, out.best_fit]
        
    """Method - iterable dictionaries {} more efficient. Names are keys and the dictionary
    contains a list. [0] refers to best fit whereas [1] would be the standard error"""
    out_dict_val_stderr = {par[i]: [out.params[par[i]].value, out.params[par[i]].stderr]
                           for i in range(len(par))}
    
    time.sleep(0.1)
    return lst_best_fit, out_dict_val_stderr

"""-------------------------lmfit dB issues---------------------------------"""
def steps_db_errors(xdata_lst, ydata_lst, tg=88, show=['Y','Y'], **kwargs):
    """
    Systematic procedure to extract parameters required for deterministic 
    benchmarking {T1, T2, sig_theta, sig_phi} from the following experiments:
        1. Free |0> => T1
        2. XX |+> => T2*
        3. YY |+> => Rot Error in degrees
        4. XXbar |+> => Phase Error in degrees

    Parameters
    ----------
    xdata_lst : list of 1D numpy arrays
        list of evolution times (in ns).
    ydata_lst : list of 1D numpy arrays
        Fidelity.
    tg : float
        gate time in ns.
    show : show report, optional
        DESCRIPTION. The default is ['Y','Y'].
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    **kwargs : dictionary
        List of optional modes.
        bool_params : list of booleans
            bool_params[0] = True or False (for A)
            bool_params[1] = True or False (for B)
            bool_params[2] = True or False (for r_clif => clifford error)
            bool_params[3] = True or False
        guess : list of float
            guess[0] => A (Initial State Probability)
            guess[1] => B (Final state probability)
            guess[2] => r_clif (clifford error)
            guess[3] => number of qubits (constant)
        weights : 1D numpy array
            examples: np.power(ydata,-1) #bias
            np.power(ydata, -0.5) # bias
            np.ones(len(ydata)) => default
        method : string
            Fitting algorithm (i.e. 'least-square', 'nelder-mead', 'differential-evolution')
        fit_res_pts : float
            Number of fitting points different from xdata
    """
    # prepare data for experiment
    step1_x, step2_x, step3_x, step4_x = xdata_lst[0], xdata_lst[1], xdata_lst[2], xdata_lst[3]
    step1_y, step2_y, step3_y, step4_y = ydata_lst[0], ydata_lst[1], ydata_lst[2], ydata_lst[3]
    
    """step 1 => determine T1 via Free; |0>. In PPT, Apply RY(pi)
    guess[0]=0, guess[1] is dummy variable, guess[2]=1
    """
    step1_fit, dict_step1 = lm_dB_errors(xdata=step1_x, 
                                         ydata=step1_y, 
                                         show=["N","N"],
                                        #  bool_params=[False, True, False],
                                        #  guess = [0, 1E-5, -1]
                                         )
    
    """step 2 => determine T2* (Ramsey T2) via XX;|+>. In PPT, Apply RY(pi/2), P1=P2=RX(pi)
    guess[0]=0, guess[1] is dummy variable, guess[2]=0"""
    step2_fit, dict_step2 = lm_dB_errors(xdata=step2_x, 
                                         ydata=step2_y, 
                                         show=["N","N"],
                                        #  bool_params=[False, True, True],
                                        #  guess = [0, 1E-5, 0]
                                         )
    
    """step 3 => determine rotation errors via YY;|+>. In PPT, Apply RY(pi/2) to |0>, P1=P2=RY(pi)
    """
    step3_fit, dict_step3 = lm_dB_errors(xdata=step3_x, 
                                         ydata=step3_y, 
                                         show=["N","N"]
                                         )
    # convert frequency to rotational errors
    sig_theta = get_db_errors(f=ufloat(nominal_value=dict_step3['f_e'][0], std_dev=dict_step3['f_e'][1]),
                              tg=tg, err_type='rot')
        
    """
    step 4 => determine rotation errors via YY;|+>. In PPT, Apply RY(pi/2) to |0>, P1=RX(pi), P2=RX(-pi)
    """
    step4_fit, dict_step4 = lm_dB_errors(xdata=step4_x, 
                                         ydata=step4_y, 
                                         show=["N","N"]
                                         )
    # convert frequency to phase errors
    sig_phi = get_db_errors(f=ufloat(nominal_value=dict_step4['f_e'][0], std_dev=dict_step4['f_e'][1]),
                            tg=tg, err_type='phase')
    # different result from 
    
    """
    step 5 => prepare dictionary of T1, T2*, sig_theta, sig_phi as well as
    best-fit models and then show report if necessary
    """
    fit_list = [step1_fit, step2_fit, step3_fit, step4_fit]
    dict_DB = {'T1_ns': [dict_step1['T_d'][0], dict_step1['T_d'][1]],
               'T2r_ns': [dict_step2['T_d'][0], dict_step2['T_d'][1]],
               'sig_theta_deg': [sig_theta.n, sig_theta.std_dev],
               'sig_phi_deg': [sig_phi.n, sig_phi.std_dev]}
    
    if show[0] == 'Y':
        print('\n')
        #this report is important as it gives measure on further analysis the option of bundling
        print(dict_DB)
        print('\n')
    
    if show[1] == 'Y':
        wfig=8.6
        fig = plt.figure(constrained_layout=True, figsize=(1*cm_to_inch(wfig),
                                                            1*cm_to_inch(wfig)))
        spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
        ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model
        
        # Data
        ax0.plot(step1_x, step1_y, 'k.', label=r'data, Free; $|0\rangle$')
        ax0.plot(step2_x, step2_y, 'b.', label=r'data, $XX$; $|+\rangle$')
        ax0.plot(step3_x, step3_y, 'g.', label=r'data, $YY$; $|+\rangle$')
        ax0.plot(step4_x, step4_y, 'm.', label=r'data, $X\bar{X}$; $|+\rangle$')
        
        # Fit
        ax0.plot(step1_fit[0], step1_fit[1], 'k-', label=r'fit, Free; $|0\rangle$')
        ax0.plot(step2_fit[0], step2_fit[1], 'b-', label=r'fit, $XX$; $|+\rangle$')
        ax0.plot(step3_fit[0], step3_fit[1], 'g-', label=r'fit, $YY$; $|+\rangle$')
        ax0.plot(step4_fit[0], step4_fit[1], 'm-', label=r'fit, $X\bar{X}$; $|+\rangle$')
        
        ax0.set_xlabel('Evolution Time (ns)')
        ax0.set_ylabel('')
        ax0.set_ylabel(r'Fidelity')
        ax0.set_ylim(-0.05, 1.05)
        ax0.legend(loc='best', ncols=2, fontsize='x-small', frameon=False)

        plt.show()
    
    return fit_list, dict_DB 

"""-------------------------lmfit.minimize version--------------------------"""
def residual_dB_errors(params, t, data=None, weights=None):
    # minimization wrapper for lmfit.minimize (for more-complicated functions)
    # this can be simplified as a wrapper. 
    
    f_e = params['f_e']
    T_d = params['T_d']
    a = params['a']
    
    # Gaussian function with offset
    model = dB_errors(tn=t, f_e=f_e, T_d=T_d, a=a)
    
    # Return residuals if data is provided, otherwise return the model
    if data is not None:
        residuals = model - data
        if weights is not None:
            residuals *= weights  # Apply the weights
        return residuals
    else:
        return model

# def lm_min_dB_errors():
    
    
#     # nelder mead does not give good agreement for Q_tls
#     out = minimize(func, params, method=fit_method[2], 
#                    args=(xdata, ydata, weights), reduce_fcn=None)
    
#     return