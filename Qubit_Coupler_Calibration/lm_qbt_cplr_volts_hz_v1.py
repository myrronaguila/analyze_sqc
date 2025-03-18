# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:21:13 2025

lmfit version of lineshapes_qbt_cplr_volts_hz. Written using Spyder IDE.
Change-log by Myrron
    2025/03/18
    1. For dissemination purposes, with the intention of having this reproduced
    on other systems.  
    2. Adapted best scripts that work with minimal redundancies due to
    code development.
    3. Reduced dependencies to minimize dependencies to other python package
    4. structure of fit models
        4.1. Model #1
            4.1.1. Guess Functions for Model 1
            4.1.2. Structural Functions for Model 1
            4.1.3. Fit Model for Model 1
        4.2. Model #1
            4.2.1. Guess function for Model 1
            4.2.2. Structural Functions for Model 1
            4.2.3. Fit Model for Model 1
    Dependencies:
    
    In the future:
        - Improved naming once all codes are confident to work.
        - relocate structural models to lm.utils or other global functions
        - Replace matplotlib.pyplot with better visualization schemes.
    
@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from lmfit import Model, Parameters, minimize
from uncertainties import ufloat

from ls_qbt_cplr_volts_hz import fj_from_f01_transmon, M_flux_Line 
from ls_qbt_cplr_volts_hz import fj_from_f01_m
from ls_qbt_cplr_volts_hz import f01_volts_to_hz  
from ls_qbt_cplr_volts_hz import anticrossing_res3

from lm_utils import show_report, find_nearest
from lm_utils import show_report_minimize

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

"""-----------------------Qubit-Frequency Dispersion Model------------------"""
def guess_f01_volts_to_hz(xdata, ydata, f_Ec=200E6):
    """
    model = (f_max + f_Ec)*np.sqrt()
    # we assume f_Ec acquired independently from two-photon excitation
        
    Parameters
    ----------
    xdata : 1D numpy array
        dc voltage / vpk (either in AU or in vpk).
    ydata : 1D numpy array
        Qubit frequency (in Hz).

    Returns
    -------
    dict_g. dictionary
        dictionary of initial guesses
    """
    # guess fq_max
    f01_max = np.amax(ydata) # for dict
    
    # guess v0
    x_offset = xdata[find_nearest(ydata, f01_max)[0]] # for dict
           
    # estimate A_conv and fact based on data range
    #get period from one minimum branch
    idx_f01_min, f01_min = find_nearest(array=ydata, value=np.amin(ydata))
    x_min = xdata[idx_f01_min]
    if x_min < x_offset:
        period = 2*(x_offset - x_min)
    else:
        period = 2*(x_min - x_offset)
    
    """
    Period of flux-tunable resonator best guessed when using a flux-tunable 
    resonator period.
    """
    A_conv_guess = 1 / period # surely off if not corresponding to one period
        
    """
    fact_guess is Ej2/Ej1. If Ej2=Ej1, then the assymmetry d=0. 
    Else Ej2/Ej1 < 0=> d<0  or Ej2/Ej1 > 1 => d>1
    # This approximate method is good from the start where assumption is fact=0
    """
    
    fact_guesses = np.linspace(0.0, 50.0, 101, endpoint=False)
    errors = [np.linalg.norm(ydata - f01_volts_to_hz(vpk=xdata, A_conv=A_conv_guess,
                                                     v0=x_offset, fact=elem))
                  for elem in fact_guesses]
    fact_guess = fact_guesses[np.argmin(errors)]
    #print(fact_guess)
    
    dict_g = {'v0': x_offset, 
              'A_conv': A_conv_guess,
              'f_max': f01_max,
              'fact': fact_guess,
              'f_Ec': f_Ec}
    return dict_g

def lm_f01_volts_to_hz(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    lmfit of qubit frequency vs flux voltage (useful for qubit and coupler 
                                              characterization)
    
    Parameters:
    
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
        guess : list of values, order according to par list 
            if guess_dict=None, we use lmfit well-defined values
        add_const : list of floats
            floats that help specify restriction in analysis        
        bool_params : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const_c)
            par_vary[1] = True or False (boolean, on exp_decay)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        axis_lbl : list of x and ylabel
            axis_lbl[0] = string: label of x-axis
            axis_lbl[1] = string: label of y-axis
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
    np.random.seed(42)
    
    """Set-up model using the build-in functions"""
    mod = Model(f01_volts_to_hz) #(vpk, A_conv, v0=0, f_max=5E9, fact=1, f_Ec=200E6)
    par = ['A_conv', 'v0', 'f_max', 'fact', 'f_Ec']
    
    # initialize parameters for simplicity of code structure
    par_var = [True, True, True, True, False]
    weights = np.ones(len(ydata))
    method=fit_method[0] #Levenberg-marquardt
    axs_lbl_fit = ['x-data', 'y-data']
    dict_g = {}
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, True, True, False] # default unique to the Model
        # to generalize, put things in default 
    
    # second version for guess; nested if-else statement for better guess
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_f01_volts_to_hz(xdata,ydata,guess[4])[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2],
                      par[3]: guess[3],
                      par[4]: guess[4]}
    else:
        dict_g = guess_f01_volts_to_hz(xdata, ydata) # better at guessing parameters, works with gpg
    
    # print(dict_g)
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    """Add Models and Evaluate"""
    #params = Parameters()
    params=mod.make_params()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], 0, 1/(xdata[1]-xdata[0])), #A_conv
                    (par[1], p[1], par_var[1], np.amin(xdata), np.amax(xdata)), #v0 
                    (par[2], p[2], par_var[2], np.amin(ydata), 2*np.amax(ydata)), #f_max
                    (par[3], p[3], par_var[3], 0, 30), #fact
                    (par[4], p[4], False))
   
    """Const_mod and Exp_Mod guess are way off but the resulting fits are quite
    robust even if the values are negative"""
    
    # this can be minimized by default values
    if 'weights' in kwargs:
        weights = kwargs['weights'] #weights, 1D numpy array
    else:
        weights = np.ones(len(ydata))
    
    if 'method' in kwargs:
        method = kwargs['method']
    else: 
        method=fit_method[0] #Levenberg-marquardt
        # method=fit_method[2] #differential errors => gives runtime errors
    
    # fit model
    out = mod.fit(ydata, params, vpk=xdata, scale_covar=True, method=method, 
                  weights=weights)
    
    """20250303 - **kwargs after model.fit. Best encapsulated in lm_utils
    for improved readability, not necessarily improved performance"""
       
    """
    'fit_res_pts' is redundant across lmfit functions. This may need to be encapsulated
    """
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) # numbers spaced in linear space
        #print(out.params)
        out_eval = out.eval(vpk=xdata_eval)
        
        # comp evals added for deconvoluted plots in case of compounded models
        comps_eval = out.eval_components(vpk=xdata_eval)
        
        # show_report, other kwargs include the following
        
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    fit_eval = [xdata_eval, out_eval, comps_eval], 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata_eval, out_eval]
    else: 
        show_report(result=out, xdata=xdata, ydata=ydata, show=show, 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata, out.best_fit]
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par[i]: [out.params[par[i]].value, 
                                    out.params[par[i]].stderr]
                            for i in range(len(par))}
    
    """Add other relevant parameters for parametrization on dictionary
    1. EJ_max
    2. d => factor
    3. Qubit Inductance Lg => Useful for Qutip Simulation
    """
    if 'device' in kwargs:
        """device = Q0, Q1, C1 => name of coupler in string"""
        out_dict_val_stderr['device']=kwargs['device']
    else:
        out_dict_val_stderr['device']='qubit'
    
    f_max = ufloat(out.params['f_max'].value, out.params['f_max'].stderr)
    fact = ufloat(out.params['fact'].value, out.params['fact'].stderr)
    A_conv = ufloat(out.params['A_conv'].value, out.params['A_conv'].stderr)
    
    # get asymmetry
    d_val = (fact - 1)/(fact + 1)
    
    # get Ej_max and error
    f_Ejmax = fj_from_f01_m(f01=f_max.n, fc=guess[4]) # for std_err
    f_Ejmax_approx = fj_from_f01_transmon(f_max, guess[4]) # for nominal value
    
    # get mutual inductance from the flux line in units of flux quanta.
    M_line = M_flux_Line(A_conv)
    
    # updated dictionary for saving, complete hamiltonian
    out_dict_val_stderr['f_Ejmax']=[f_Ejmax, f_Ejmax_approx.std_dev] 
    out_dict_val_stderr['d']=[d_val.n, d_val.std_dev]
    out_dict_val_stderr['M_line']=[M_line.n, M_line.std_dev] # for qutip conversion
    
    time.sleep(0.1)
    return lst_best_fit, out_dict_val_stderr

"""-----------------------Anti-Crossing Models------------------------------"""
def guess_anticrossing_model(xdata, ydata, show=['N','N'], **kwargs):
    """
    Get precisely the idling frequency and coupling strength.

    Parameters
    ----------
    xdata : 1d np array
        tunable qubit frequency.
    ydata : 1d np array
        dressed-state frequency (with anti-crossings).

    Returns
    -------
    dictionary: guess values
    """
    # initial guess for f_fixed    
    f_fixed = np.average(ydata) # get average of splitted data for idle guess
        
    """
    20250313 - data measurements
    Better guess for idle-frequency in between upper and lower branches 
    can be done in three ways:
        1. Median-based estimation (assuming symmetric spectrum)
        2. Moving average (assuming noisy data)
        3. k-means clustering (uneven, missing points)
    For the measurement method, K-means clustering could be the best bet.
    
    Why bother spending time with this? -> even if resonator and fqidle are
    known, we at least know how to get the parameters right even without
    raw data. Though the real measured data would be reliable. Moreover,
    there is an issue of qubit / coupler frequency drift.
    """

    # **Step 1: Initial Estimate for f_fixed (Use Median)**
    f_fixed = np.median(ydata)
    
    # **2. Apply percentile-based filtering**
    upper_mask = ydata >= f_fixed
    lower_mask = ydata < f_fixed

    # Get **20th percentile for upper branch** (ignores saturation)
    upper_min = np.percentile(ydata[upper_mask], 20) if np.any(upper_mask) else np.min(ydata)
    #print(upper_min) #4767570000.0

    # Get **80th percentile for lower branch** (ignores extreme sparse data)
    lower_max = np.percentile(ydata[lower_mask], 80) if np.any(lower_mask) else np.max(ydata)
    #print(lower_max) #4759150000.0

    # **3. Compute final estimate**
    f_fixed_opt = 0.5 * (upper_min + lower_max)
    
    # get g from the distance between a y-value near x_anti and the f_fixed_opt
    idx_xanti, x_anti = find_nearest(array=xdata, value=f_fixed_opt)
    g_hz = np.abs(ydata[idx_xanti] - f_fixed_opt)
        
    dict_g = {'f_fixed': f_fixed_opt,
              'g_hz': g_hz}
    
    if show[0]=='Y':
        print('\n')
        print('Init f_fixed = {:.6e} Hz'.format(f_fixed))
        print('Guess f_fixed={:.6e} Hz'.format(f_fixed_opt))
        print('Guess g_hz = {:.6e} Hz'.format(g_hz))
    
    if show[1]=='Y':
        colors = np.where(ydata > f_fixed_opt, "red", "blue")
        
        plt.figure(figsize=(8, 5))
        plt.scatter(xdata, ydata, c=colors, alpha=0.6, label="Data")
        plt.axhline(f_fixed, color="blue", linestyle="--", label=f"Median Estimate = {f_fixed:.2f}")
        plt.axhline(f_fixed_opt, color="green", linestyle="--", label=f"Refined Estimate = {f_fixed_opt:.2f}")
        plt.xlabel("Control Parameter (x)")
        plt.ylabel("Frequency (f1)")
        plt.title("Refined Estimate of f_fixed Using Percentile Filtering")
        plt.legend()
        plt.show()
    
    return dict_g

def sort_anticrossing_data(f1_combined, y_combined, branch_labels):
    """
    If the upper and lower branches are completely disjoint 
    (with no common f1 points), sorting f1_combined before fitting can improve 
    visualization:
    
    # best transferred to lm_utils    
    """
    sort_indices = np.argsort(f1_combined)
    return f1_combined[sort_indices], y_combined[sort_indices], branch_labels[sort_indices] 

def struct_anticrossing_data(xdata, ydata, y_idle):
    """
    Restructure anticrossing data for lm-minimization

    Parameters
    ----------
    xdata : 1d numpy array
        frequency of coupler (can be double-valued.
    ydata : 1d numpy array
        frequency of qubit.

    Returns
    -------
    f1_combined : TYPE
        DESCRIPTION.
    y_combined : TYPE
        DESCRIPTION.
    branch_labels : TYPE
        DESCRIPTION.

    """
    # Assuming xdata and ydata contain mixed upper/lower points

    # Identify which points belong to upper and lower branches
    upper_branch_indices = ydata > y_idle  # Roughly separate
    lower_branch_indices = ~upper_branch_indices

    # Separate x-data and y-data explicitly
    f1_upper, f1_lower = xdata[upper_branch_indices], xdata[lower_branch_indices] 
    y_upper, y_lower = ydata[upper_branch_indices], ydata[lower_branch_indices]

    # Combine them, allowing redundant f1 values
    f1_combined = np.concatenate([f1_upper, f1_lower])
    y_combined = np.concatenate([y_upper, y_lower])
    branch_labels = np.array(['up'] * len(f1_upper) + ['down'] * len(f1_lower))
    
    # return f1_combined, y_combined, branch_labels # for unsorted data
    
    # sort anti-crossing data according to order, more efficient fit. 
    return sort_anticrossing_data(f1_combined, y_combined, branch_labels)

def lm_min2_anticrossing(xdata, ydata, show=['Y','Y'], **kwargs):
    np.random.seed(42) 
    """   
    2025/03/18
    lmfit of anticrossing interaction to best-fit anticrossing spectra
    between one quantum element (resonator, qubit) and coupler (artificial atom,
    mechanical resonator, NEMs)
    
    Considers labelling of upper and lower branches for effective measurement
    preserves 1D numpy array for improved fitting.
    xdata contains duplicate upper branch, and lower branch (suggested sequence).
    ydata has both upper branch and lower branch
    
    Note: The estimation of the idle frequency is near that of literature but 
    the coupling strength may not be optimized. Its best fit of the coupling
    strength works best if the only free parameter is the coupling strength.
    
    Parameters:
    
    xdata : 1D npy array
        voltage (V).
    ydata : 1D np array
        tuning qubit/coupler/NEMS frequency.
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        guess : list of values, order according to par list 
            if guess_dict=None, we use lmfit well-defined values
        add_const : list of floats
            floats that help specify restriction in analysis        
        bool_params : list of booleans
            default = [True, True, True]
            par_vary[0] = True or False (boolean, on const_c)
            par_vary[1] = True or False (boolean, on exp_decay)
            par_vary[2] = True or False (boolean, on exp_amplitude)
        axis_lbl : list of x and ylabel
            axis_lbl[0] = string: label of x-axis
            axis_lbl[1] = string: label of y-axis
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
    lst_best_fit. list of 1D numpy array
        lst_best_fit[0] => x-axis array
        lst_best_fit[1] => y-axis array
    out_dict_values:  dictionary of results
        out_dict_values[params]=[nominal values, std_error]
    """
    
    """Set-up minimization using anti-crossing functions"""
    params = Parameters()
    par = ['f_fixed', 'g_hz']
    
    # initialize parameters for simplicity of code structure
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True] # default unique to the Model
        # to generalize, put things in default 
    
    dict_g = {}
    axs_lbl_fit = ['x-data', 'y-data']
    
    # second version for guess; nested if-else statement for better guess
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_anticrossing_model(xdata,ydata,['N','N'])[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1]}
    else:
        dict_g = guess_anticrossing_model(xdata, ydata,['N','N']) # better at guessing parameters, works with gpg
       
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    """Add Models and Evaluate"""
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], np.amax(ydata[ydata<p[0]]), 
                     np.amin(ydata[ydata>p[0]])), #A_conv
                    (par[1], p[1], par_var[1], 1E6, 1E9))
    
    # Fit range too dependant on p[0]. Best to set p[0] by eye rather than np.average.
   
    """Const_mod and Exp_Mod guess are way off but the resulting fits are quite
    robust even if the values are negative"""
    
    # this can be minimized by default values, performant than pythonic info
    if 'weights' in kwargs:
        weights = kwargs['weights'] #weights, 1D numpy array
    else:
        weights = np.ones(len(ydata))
    
    if 'method' in kwargs:
        method = kwargs['method']
    else: 
        method=fit_method[0] #Levenberg-marquardt
    
    # simplification: eats some performance at the expense of readability.
    # weights = kwargs.get('weights', np.ones(len(ydata)))
    # method = kwargs.get('method', fit_method[0])
    
    """sort xdata and ydata for minimization according to upper and lower branches"""
    
    xdata_br, ydata_br, branch_br = struct_anticrossing_data(xdata, ydata, p[0])    
    """Notes on the sort method, sort method do not have pointers to remind
    readers on the branch data when using it. 
    """
    
    # inject weights ot improve fit
    upper_weight = 1 / np.sum(branch_br == 'up')
    lower_weight = 1 / np.sum(branch_br == 'down')

    weights = np.where(branch_br == 'up', upper_weight, lower_weight)
    
    # provide sorted branch
    # minimize residuals
    out = minimize(
        anticrossing_res3, 
        params, 
        method=method,
        scale_covar=True,
        args=(xdata_br, ydata_br, branch_br, weights, 'res')
    )
    
    """Version 2 - ordering"""
    
    # show best fit curve according to sorted branch, sort branch for visualization
    branch_br = np.array(['up' if ydata[i] > p[0] else 'down' 
                         for i in range(len(ydata))])
    
    out_best_fit = anticrossing_res3(out.params, f_tune=xdata, data=ydata, 
                                     branch= branch_br, 
                                     weights=weights,
                                     output='model')
    init_fit = anticrossing_res3(out.params, f_tune=xdata, data=ydata,
                                 branch= branch_br,
                                 weights=weights,
                                 output='model')
    
    if 'fit_res_pts' in kwargs:
        # here, it is false
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        init_eval_fit = anticrossing_res3(out.params, f_tune=xdata, data=ydata, 
                                          branch= branch_br, weights=weights,
                                          output='model')
        out_eval_fit = anticrossing_res3(out.params, n=xdata_eval, data=ydata, 
                                         branch= branch_br, weights=weights,
                                         output='model')
        # no component-fit for this fit report
        show_report_minimize(result=[out, init_eval_fit, out_best_fit], 
                             xdata=xdata, ydata=ydata, show=show, 
                             fit_eval = [xdata_eval, out_eval_fit], 
                             axis_label = axs_lbl_fit, **kwargs)
        
        # list best fit model
        lst_best_fit = [xdata_eval, out_eval_fit]
    else: 
        # most active data
        show_report_minimize(result=[out, init_fit, out_best_fit], xdata=xdata, 
                             ydata=ydata, show=show, 
                             axis_label = axs_lbl_fit, **kwargs)
        
        # output best fit model
        lst_best_fit = [xdata, out_best_fit]
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par[i]: [out.params[par[i]].value, 
                                    out.params[par[i]].stderr]
                            for i in range(len(par))}
    
    # label device in kwargs
    if 'device' in kwargs:
        """device = Q0, Q1, C1, R1, etc => name of coupler in string"""
        out_dict_val_stderr['device']=kwargs['device']
    else:
        out_dict_val_stderr['device']='qubit'
    
    # simplification # if else is faster than kwargs.get
    out_dict_val_stderr['device'] = kwargs.get('device', 'qubit')
        
    time.sleep(0.1)
    
    return lst_best_fit, out_dict_val_stderr


