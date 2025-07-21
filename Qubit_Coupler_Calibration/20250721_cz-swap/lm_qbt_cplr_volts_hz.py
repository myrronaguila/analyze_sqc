# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:38:09 2025

lmfit for spectroscopy of coupler
Updated: 20250303 -> github


@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from data_plots_qm import cm_to_inch
from scipy.stats import linregress
from lmfit import Model, Parameters, minimize, Minimizer
from uncertainties import ufloat

from qbt_cplr_volts_hz import f01_volts_to_hz, fj_from_f01_m, d_to_fact
from qbt_cplr_volts_hz import fj_from_f01_transmon, M_flux_Line
from qbt_cplr_volts_hz import anticrossing_model, anticrossing_res2
from qbt_cplr_volts_hz import anticrossing_res3
from qbt_cplr_volts_hz import signed_cos # dealing with negative sign

from lm_utils import show_report, find_nearest
from lm_utils import show_report_minimize
# 20250321 - added format capabilities for save
from lm_utils import format_data_ctrl_dics, save_fit_results
from lm_utils import kwargs_save

# 20250515
from qbt_cplr_volts_hz import g_qq_hz_res1, g_qq_hz_res2

from lm_tdm import lm_wave_c, lm_wave_c2 # useful in extracting 

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

"""--------------------------lmfit-transmon model---------------------------"""
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

def optimize_fmax(xdata, ydata, f_Ec=200E6):
    """Optimize f_max for better guess extraction
    - algorithm based on greedy optimizers
    """
    
    # set-up objective function
    def objective(params, ydata, f_Ec):
        f_max = params['f_max']
        
        # Compute y_adj
        y_adj = np.power((ydata + f_Ec) / (f_max + f_Ec), 4) #<=1
        
        # We want the maximum of y_adj to be as close as possible to 1
        max_y_adj = np.max(y_adj)
        return max_y_adj - 1
    
    params = Parameters()
    params.add('f_max', value=1.0, min=np.max(ydata))
    
    # Perform optimization
    minimizer = Minimizer(objective, params, fcn_args=(ydata, f_Ec))
    result = minimizer.minimize(method='nelder')
    
    # Optimized f_max
    f_max_opt = result.params['f_max'].value
    return f_max_opt

def optimize_gam(xdata, ydata, A_conv, v0, f_max, f_Ec=200E6):
    """Optimize fact for better guess extraction
    - algorithm based on greedy optimizers
    """
    
    # set-up objective function
    def objective(params, ydata, v0, f_max, f_Ec):
        fact = params['fact']
        A_conv = params['A_conv']
        
        # Compute y_adj
        y_model = f01_volts_to_hz(xdata, A_conv, v0, f_max, fact, f_Ec) 
        # print(fact) 
        return np.sum((y_model - ydata) ** 2)  # Residual sum of squares
    
    params = Parameters()
    params.add('fact', value=0.5, min=0, max=10.0)
    params.add('A_conv', value=A_conv, min=1/(np.amax(xdata)-np.amin(xdata)), max=1/(xdata[1]-xdata[0]))
    
    
    # Perform optimization
    minimizer = Minimizer(objective, params, fcn_args=(ydata, v0, f_max, f_Ec))
    result = minimizer.minimize(method='nelder')
    # result = minimizer.minimize(method='differential_evolution')
    # Optimized f_max
    fact_opt = result.params['fact'].value
    # A_conv_opt = result.params['A_conv'].value
    # print('A_conv_opt')
    # print(A_conv_opt)
    return fact_opt

def guess_f01_volts_to_hz_v2(xdata, ydata, f_Ec=200E6, show=['N', 'N']):
    """
    This guess method turned out to be too unstable though the idea of fitting
    to sine-wave is quite robust. The core issue lies with the optimization function.
    
    -> uses the following algorithms: 

    Parameters
    ----------
    xdata : TYPE
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.
    f_Ec : TYPE, optional
        DESCRIPTION. The default is 200E6.
    show : TYPE, optional
        DESCRIPTION. The default is ['Y', 'Y'].

    Returns
    -------
    dict_g : TYPE
        DESCRIPTION.

    """
    # we do greedy optimization to optimize critical parameter of 'f' -> open-loop optimization
    
    # start of section for optimization for ver1
    
    # # guess fq_max
    f01_max = np.amax(ydata) # for dict # rough but not ideal
    #print(f01_max)
        
    """
    Period of flux-tunable resonator best guessed when using a flux-tunable 
    resonator period.
    """
    
    # optimize f01_max based on lmfit.minimize, get optimized amplitude
    f01_max = optimize_fmax(xdata, ydata, f_Ec=f_Ec)
    #print(f01_max) #5424730000.0 > f_max_apl_sci = 5423.5E6
    
    # use cosine wave with offst to get A_conv and fact # fact is better than d since 0 < fact <= np.inf.
    y_adj = np.power((ydata + f_Ec)/(f01_max + f_Ec),4)
    
    # fit y_adj to extract parameters that can be converted to data.
    y_adj_fit, guess_dict = lm_wave_c2(xdata=xdata, ydata=y_adj, show=show, 
                                         wave='cos')
    #print(guess_dict)
    A_conv_guess = guess_dict['cos_frequency'][0]/2 # 2*A_conv in the guess_cos_freq
    x_offset = - guess_dict['cos_shift'][0]/guess_dict['cos_frequency'][0] # fit shows positive phase
    
    # fact_guess = d_to_fact(np.sqrt(2*guess_dict['const_c'][0] - 1)) #const_c = (1 + d**2)/2 => d**2 = 2*const_c-1
    
    # use optimizer
    
    # fact_guess = optimize_gam(xdata, ydata, A_conv_guess, x_offset, f01_max, f_Ec)
    # the tricky issue to make d**2. d**2= (\gamma - 1)**2/(\gamma + 1)**2
    fact_guess = 1 # good assumption as converting d is complicated. Make d free parameter
    
    """
    Quite robust. values with minimal errors even for an incomplete wave. 
    Idea lies with the simplicity of sinusoidal nature of the resulting
    amplitude. Key goal is to capture the frequency, shift, amplitude and const_c
    for useful parameters for the non-linear fit.
    
    (y_adj)^4 = A + B cos(C*V + phi)
    d = (y-1)/(y+1); d^2 = (y-1)^2 / (y+1)^2;  1-d^2 = ((y^2 + 2y + 1) - (y^2 - 2y + 1))/(y+1)^2 ; 4y/((y+1)^2)
    (1-d^2)/2 = 2y/((y+1)^2)
    
    A = (1 + d^2)/2 = (y^2 + 1)/((y+1)^2)
    B = (1 - d^2)/2 = 2y/((y+1)^2)
    A - B = d^2 = (y-1)^2/(y + 1)^2 => const_c + cos_amp < 1 # within error bar
    
    # d is a parameter that varies with multiplicative inverse. We use the B conversion
    # key idea. In principe, cnst_c and cos_amp should have positive. 
    choose for values of A and d in a way that yields positive values.
    
    B=0.52*2 = 1.04=(1-d^2)=> d^2 = -0.04
    A=0.48*2 = 1 + d^2 => d^2 =-0.04. 
    a parametric conundrum. => simple solution. Wrong scaling.
    
    y=(1/cos_amp)+/-np.sqrt((1/cos_amp)^2 - 4)
    y^2 - (2/cos_amp)*y + 1 = 0; y=(-2/cos_amp +/- np.sqrt((2/cos_amp)^2 - 4*1*1))/(2)
    y=(-1/cos_amp)+/-np.sqrt((1/cos_amp)^2 - 4)
    
    20250220 -> result
    const_c:        0.47964779 +/- 0.01853375 (3.86%) (init = 0.8857997)
    cos_amplitude:  0.51994646 +/- 0.01840886 (3.54%) (init = 0.1349289)
    cos_frequency:  3.83412576 +/- 0.07527006 (1.96%) (init = 10.29419)
    cos_shift:      0.01805738 +/- 5.4811e-04 (3.04%) (init = 0.06220976)
    
    B=0.52*2 = 1.04=(1-d^2)=> d^2 = -0.04
    A=0.48*2 = 1 + d^2 => d^2 =-0.04. 
    
    20250221 -> after constraint with lm_wave_c2 (with constrained _c)
    
    const_c:        0.49698683 +/- 0.01461518 (2.94%) (init = 0.8864848)
    cos_amplitude:  0.50301317 +/- 0.01461518 (2.91%) == '1 - const_c'
    cos_frequency:  3.89577877 +/- 0.06182061 (1.59%) (init = 10.29419)
    cos_shift:      0.01835913 +/- 5.1870e-04 (2.83%) (init = 0.06220976)
    
    B=0.503*2 = 1.006 = (1 - d^2) => d^2 = -0.04
    A=0.497*2 = 0.994 = (1 + d^2) => d^2 =-0.04.
    A > B. => issue is that f_max is not optimized but this is a good start
    
    A + B = 1
    A - B = d^2
    
    # intention of the fit is to extract cos_freq and cos_shift => Ve.
    A_conv = cos_frequency => get
    # 
    
    Once obtained. Fix the cos_frequency and shift. 
    We then use the constant frequency and shift to calibrate the following:
        f_max, and d. Once this is done. The corrected d will be easier to extract.
        
    Idea is that A + B = 1. If not 1, then there is a fit problem.
    """
        
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
        # xdata_eval = np.geomspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
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
    
    """---------------Extract derived info from output devices--------------"""
    f_max = ufloat(out.params['f_max'].value, out.params['f_max'].stderr)
    fact = ufloat(out.params['fact'].value, out.params['fact'].stderr)
    A_conv = ufloat(out.params['A_conv'].value, out.params['A_conv'].stderr)
    
    # get asymmetry
    d_val = (fact - 1)/(fact + 1)
    
    # get Ej_max and error
    f_Ejmax = fj_from_f01_m(f01=f_max.n, fc=guess[4]) # for std_err
    f_Ejmax_approx = fj_from_f01_transmon(f_max, guess[4]) # for nominal value
    
    # get mutual inductance from the flux line
    M_line = M_flux_Line(A_conv)
    
    # updated dictionary for saving, complete hamiltonian
    out_dict_val_stderr['f_Ejmax']=[f_Ejmax, f_Ejmax_approx.std_dev] 
    out_dict_val_stderr['d']=[d_val.n, d_val.std_dev]
    out_dict_val_stderr['M_line']=[M_line.n, M_line.std_dev] # for qutip conversion
    
    """Format data_dic and ctrl_dic for save functions"""
    
    """Pull option for save function of fitting methods, and parameters
    for reproducibility."""
    
    if 'save' in kwargs:
        # ver1. A must function for utility-grade fitter
        # ['param_names', 'bool_params', 'guess', 'method', 'weights']
        # ver2, note that all variables must not change.
        kwargs_save(fit_val_dic=out_dict_val_stderr, xdata=xdata, ydata=ydata,
                    fit=lst_best_fit, param_names=par, bool_params=par_var,
                    guess=p, method=method, weights=weights, save=kwargs['save'])
        # no need for optimization.
        
    time.sleep(0.1) # delay for autofit functions
    return lst_best_fit, out_dict_val_stderr

"""
Note 20250303 - in improving modelling of fit errors, one could consider
a kwargs argument before model fit and after model fit for better lm_model 
prototyping like:
    1. kwargs_before_model (except guess which is sensitive to modifications)
        1.1. 'bool_params'
        1.2. 'weights'
        1.3. 'method'
        1.4. 'axis_label'
        1.5. 'device'
    2. kwargs_after_model
        2.1. 'fit_res_pts' => perhaps optional 
        => core-issue is that the independent variable varies per model.
        for this kwargs to work, one needs to get the independent variable
        from out.eval
        2.2. 'device' => optional
Improving readability of fit model
"""


"""lmfit-two-tone anti-crossing model (for resonator and idling qubit freq)---"""


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
    
    # # 20250312 - working with kmeans
    # # # Identify indices far from the anticrossing # assuming crossing is on the middle
    # far_from_crossing = np.abs(xdata - np.median(xdata)) > np.std(xdata)

    # # Extract upper and lower branches in those regions 
    # upper_far = ydata[far_from_crossing & (ydata > np.median(ydata))]
    # lower_far = ydata[far_from_crossing & (ydata < np.median(ydata))]
    # x_upper_far = xdata[far_from_crossing & (ydata > np.median(ydata))]
    # x_lower_far = xdata[far_from_crossing & (ydata < np.median(ydata))]
    
    # # Interpolate lower branch to match upper branch points
    # interp_lower = interp1d(x_lower_far, lower_far, kind='linear', fill_value="extrapolate")
    # lower_far_interp = interp_lower(x_upper_far)  # Interpolated lower branch

    # # Estimate f2_0 as the mean of the midpoints
    # f_fixed = np.mean((upper_far + lower_far_interp) / 2)
    
    # interpolation losses with longer ranges. not also accurate measurement
    
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
    
    #A = np.amin(ydata[ydata > f_fixed]) - np.amax(ydata[ydata < f_fixed])
    #B = np.amax(ydata[ydata > f_fixed]) - np.amin(ydata[ydata < f_fixed])
    #g_hz = (A+B)/2
    #print((np.amax(ydata[ydata > f_fixed]) - np.amin(ydata[ydata < f_fixed]))/2)
        
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

    # best transferred to lm_utils
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
    
    # enables global optimization based on results
    
    # return f1_combined, y_combined, branch_labels # for unsorted data
    
    # sort anti-crossing data according to order, more efficient fit. 
    return sort_anticrossing_data(f1_combined, y_combined, branch_labels)

def lm_anticrossing_model(xdata, ydata, show=['Y','Y'], **kwargs):
    """
    lmfit of interaction between tunable atom and idling resonator (i.e. resonator,
                                                                    qubit, coupler)
    
    According to LCH, we need fits like this. Strategy, use lm_anticrossing_model
    to approximate g (priority) and f_fixed (and then use these as optimizer)
    for the dressed-state spectrum with coupler.
    
    -> model only useful when model has a one-to-one mapping between x and y-variabls
    
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
    lst_best_fit. list of 1D numpy array
        lst_best_fit[0] => x-axis array
        lst_best_fit[1] => y-axis array
    out_dict_values:  dictionary of results
        out_dict_values[params]=[nominal values, std_error]
    """
    np.random.seed(42) # requirement, well structured data, xdata=N, ydata=2N
    
    # note: test data possesses elements with upper branch, lower branch and upper branch, sequentially
    # best use minimize
    
    """Set-up model using the build-in functions"""
    mod = Model(anticrossing_model) #(fq_fixed, g_hz)
    par = ['f_fixed', 'g_hz']
    
    # initialize parameters for simplicity of code structure
    par_var = [True, True]
    weights = np.ones(len(ydata))
    method=fit_method[0] #Levenberg-marquardt
    axs_lbl_fit = ['x-data', 'y-data']
    dict_g = {}
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True] # default unique to the Model
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
                    dict_g[par[i]] = guess_anticrossing_model(xdata,ydata)[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1]}
    else:
        dict_g = guess_anticrossing_model(xdata, ydata) # better at guessing parameters, works with gpg
    
    # print(dict_g)
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    """Add Models and Evaluate"""
    #params = Parameters()
    params=mod.make_params()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], np.amin(ydata), np.amax(ydata)), #A_conv
                    (par[1], p[1], par_var[1], 0, 
                     (np.amax(ydata[ydata>p[0]])-np.amin(ydata[ydata<p[0]]))/2))
   
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
    out = mod.fit(ydata, params, f_tune=xdata, scale_covar=True, method=method, 
                  weights=weights)
    
    """20250303 - **kwargs after model.fit. Best encapsulated in lm_utils
    for improved readability, not necessarily improved performance"""
       
    """
    'fit_res_pts' is redundant across lmfit functions. This may need to be encapsulated
    """
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) # numbers spaced in linear space
        # xdata_eval = np.geomspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
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
        """device = Q0, Q1, C1, R1, etc => name of coupler in string"""
        out_dict_val_stderr['device']=kwargs['device']
    else:
        out_dict_val_stderr['device']='qubit'
        
    time.sleep(0.1)
    return lst_best_fit, out_dict_val_stderr

def lm_min_anticrossing(xdata, ydata, show=['Y','Y'], **kwargs):
    np.random.seed(42) # requirement, well structured data, xdata=N, ydata=N
    
    """It works but drawback of this approach comes when there are duplicate
    frequency data between upper branch/es and lower branches."""
    
    """Set-up minimization using anti-crossing functions"""
    
    # version 1, not preferred
    
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
                    dict_g[par[i]] = guess_anticrossing_model(xdata,ydata)[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1]}
    else:
        dict_g = guess_anticrossing_model(xdata, ydata) # better at guessing parameters, works with gpg
    
    # print(dict_g)
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    """Add Models and Evaluate"""
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # A = (np.amax(ydata[ydata>p[0]])-np.amin(ydata[ydata<p[0]]))/2
    # print(A) # bound was badly chosen
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], np.amax(ydata[ydata<p[0]]), 
                     np.amin(ydata[ydata>p[0]])), #A_conv
                    (par[1], p[1], par_var[1], 1E6, 1E9))
   
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
    
    # minimize residuals
    out = minimize(
        anticrossing_res2, 
        params, 
        method=method,
        scale_covar=True,
        args=(xdata, ydata, weights, 'res')
    )
        
    # residual version
    out_best_fit = anticrossing_res2(out.params, f_tune=xdata, data=ydata, weights=weights,
                                    output='model')
    init_fit = anticrossing_res2(out.params, f_tune=xdata, data=ydata, weights=weights,
                                output='model')
    
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        init_eval_fit = anticrossing_res2(out.params, f_tune=xdata, data=ydata, weights=weights,
                                         output='model')
        out_eval_fit = anticrossing_res2(out.params, n=xdata_eval, data=ydata, weights=weights,
                                        output='model')
        # no component-fit for this fit report
        show_report_minimize(result=[out, init_eval_fit, out_best_fit], xdata=xdata, ydata=ydata, show=show, 
                    fit_eval = [xdata_eval, out_eval_fit], 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata_eval, out_eval_fit]
    else: 
        show_report_minimize(result=[out, init_fit, out_best_fit], xdata=xdata, ydata=ydata, show=show, 
                    axis_label = axs_lbl_fit, **kwargs)
        lst_best_fit = [xdata, out_best_fit]
    
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
        """device = Q0, Q1, C1, R1, etc => name of coupler in string"""
        out_dict_val_stderr['device']=kwargs['device']
    else:
        out_dict_val_stderr['device']='qubit'
        
    time.sleep(0.1)
    # problem here are the guess function. It is off.
    
    return lst_best_fit, out_dict_val_stderr

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
    
    # simplification: eats some performance.
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
        init_eval_fit = anticrossing_res3(params, f_tune=xdata, data=ydata, 
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

"""------------------------Minimized Dressed Eigenvalues using Qutip--------
Current fitting models allow independent extraction of coupling strength
and qubit frequency of couplers. The minimum requirement of getting coupling
strength from couplers are also attained. Using dressed state using qutip
is so far towards refine-ment in the fit. Better do this last for rapid
development of fitting algorithms (i.e. bridge characterization.)

Suggested sequences of minimization: rather than simultaneous fit:
    => towards scalable automated calibration in seconds.
    1. Extract coupler frequency and parabolic frequency from spectrum for
    quick fit (may need automation for extracting the bare state)
    2. Minimize iSWAP coupling. Assuming known idle frequency, and guess 
    coupling for robustness.
    3. Use qutip + lmfit for overall dressed state and then bare-state 
    minimization.
    4. Create streamlined dictionary for saving parameters with certain date
    (in-preparation for X-arrays.)
    (rather than we adapt multi-objective minimization but going towards
     greedy algorithm (series of 1D plots))
"""

"""-----------20250421------LM-signed cosine fit----------------------------"""
def guess_signed_cos(xdata, ydata, show=['Y','Y'], **kwargs):
    # guess ofs
    ofs_g = np.mean(ydata)
    
    # guess frequency and amplitude
    y_adj = ydata - ofs_g
    frequencies = np.fft.fftfreq(len(xdata), abs(xdata[-1] - xdata[0]) / (len(xdata) - 1))
    fft = abs(np.fft.fft(y_adj))
    argmax = abs(fft).argmax()
    amp_g = 2.0 * fft[argmax] / len(fft)
    freq_g = 2*np.pi * frequencies[argmax]
    
    # guess phase_g
    pts_guess = 101 #improved but slow
    shift_guesses = np.linspace(0, 2*np.pi, pts_guess, endpoint=False)
    errors = [np.linalg.norm(signed_cos(xdata, freq_g, shift_guess, amp_g, ofs_g) - ydata)
              for shift_guess in shift_guesses]
    # getting the best residual from linalg.norm
    
    phase_g = shift_guesses[np.argmin(errors)]
    
    dict_g = {'freq': freq_g, 
              'phase': phase_g,
              'amp': amp_g,
              'ofs': ofs_g}
    return dict_g

def lm_signed_cos(xdata, ydata, show=['Y', 'Y'], **kwargs):
    np.random.seed(42)
    # not useful
    
    """Set-up model using the build-in functions"""
    mod = Model(signed_cos) #(vpk, A_conv, v0=0, f_max=5E9, fact=1, f_Ec=200E6)
    par = ['freq', 'phase', 'amp', 'ofs']
    
    # initialize parameters for simplicity of code structure
    par_var = [True, True, True, True]
    weights = np.ones(len(ydata))
    method=fit_method[0] #Levenberg-marquardt
    axs_lbl_fit = ['x-data', 'y-data']
    dict_g = {}
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, True, True] # default unique to the Model
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
                    dict_g[par[i]] = guess_signed_cos(xdata,ydata)[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2],
                      par[3]: guess[3]}
    else:
        dict_g = guess_signed_cos(xdata, ydata) # better at guessing parameters, works with gpg
    
    # print(dict_g)
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    
    """Add Models and Evaluate"""
    #params = Parameters()
    params=mod.make_params()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], 0), #freq
                    (par[1], p[1], par_var[1], -2*np.pi-1.e-5, 2*np.pi+1.e-5), #phase
                    (par[2], p[2], par_var[2], 0), #amp
                    (par[3], p[3], par_var[3], np.amin(ydata), np.amax(ydata))) #ofs
   
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
    out = mod.fit(ydata, params, x=xdata, scale_covar=True, method=method, 
                  weights=weights)
    
    """20250303 - **kwargs after model.fit. Best encapsulated in lm_utils
    for improved readability, not necessarily improved performance"""
       
    """
    'fit_res_pts' is redundant across lmfit functions. This may need to be encapsulated
    """
    if 'fit_res_pts' in kwargs:
        # role of fit_res_pts - must be logspace rather than linspace because of huge magnitudes
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) # numbers spaced in linear space
        # xdata_eval = np.geomspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
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
    
    time.sleep(0.1) # delay for autofit functions
    return lst_best_fit, out_dict_val_stderr

"""--------------------Coupling Strength for CZ and iSWAP-------------------"""
def guess_g_qcq_hz1(xdata, ydata, f1, gate='iswap', 
                    dict_fc={}, show=['Y','Y'], num_fitfrac = 20, **kwargs):
    """
    Simplified data based on simplified guess. Can be used as upper bound for
    guess function. Interesting to include the effect of zz_g but let us not
    include them.

    Parameters
    ----------
    xdata : 1D numpy array
        voltage.
    ydata : 1D numpy array
        g-value (in hz).
    f1 : float
        operating frequency (in hz).    
    gate : TYPE, optional
        DESCRIPTION. The default is 'cz'.
    dict_fc : dictionary
        dictionary of converting biases to transmon frequencies
    num_fitfrac : int
        number of precision for linear regression fit. default = 20
    
    **kwargs : 1D numpy array
        dictionary.

    Returns
    -------
    dict_g : dictionary
        contains 'g_ic', 'g_qq', 'K'

    """
    
    x_fc = f01_volts_to_hz(vpk=xdata, A_conv=dict_fc['A_conv'],
                           v0=dict_fc['v0'], 
                           f_max = dict_fc['f_max'], 
                           fact=dict_fc['fact'], 
                           f_Ec=dict_fc['f_Ec'])
    
    """use correction constant for guess based on gates"""
    if gate=='iswap':
        const=1
    else:
        const=np.sqrt(2)
    
    # Sort by increasing wc
    sorted_indices = np.argsort(x_fc)
    x_fc_sorted = x_fc[sorted_indices]
    #print(x_fc_sorted) # lowest frequency of x_fc to highest frequency of x_fc
    # correct magnitude of y-sorted based on quantum gate being swept
    y_sorted = ydata[sorted_indices] / const

    """
    Linear approximation: y ≈ const*(gqq - 2*gprod*X)
    where X = wc / (wc^2 - w1^2), 
    then y = m*X+b with m = -2*gprod, b=gqq,
    for better linearity, we iterate the minimum fit_fraction
    available for the guess, based on linearity rval, not 
    the most minimum g_qq.
    """
    # use fit_fractions to select region of linearity, also considers outliers.
    fit_fractions = np.linspace(0.05, 1.0, num_fitfrac)
    slopes = np.ones(num_fitfrac)
    intercepts = np.ones(num_fitfrac)
    rvals = np.ones(num_fitfrac)
    Ns = np.ones(num_fitfrac)
    for i in range(len(fit_fractions)):
        # Select high-frequency tail (wc >> w1)
        N = int(len(x_fc_sorted) * fit_fractions[i])
        x_fc_tail = x_fc_sorted[-N:]
        y_tail = y_sorted[-N:]
        
        # do linear regression of tails and extract r_val
        X = x_fc_tail / (np.power(x_fc_tail, 2)-np.power(f1,2))
        slopes[i], intercepts[i], rvals[i], *_ = linregress(X, y_tail)
        rvals[i] = np.abs(rvals[i])
        Ns[i] = N
    
    # get idx of maximum rvals, pearson coefficient
    idx_lin = np.where(rvals == np.amax(rvals))[0][0]
    slope = slopes[idx_lin]
    intercept = intercepts[idx_lin]
    Ni = int(Ns[idx_lin])
        
    # guess values
    # determines gqq_g
    gqq_g = intercept
    # determine g_ic
    gprod_g = -slope / 2
    gic_g = np.sqrt(gprod_g)
    """
    determine renormalization factor => useful as one can plug it in ranges
    from g_ic_g >> g_qq_g and one can put ranges from 1 to 20 with better
    parameter control than gic_g. Good first guess for the dataset
    """
    K_g = gic_g / gqq_g if gqq_g != 0 else 20.0  # fallback
    
    dict_g = {'gic_hz': gic_g,
              'K': K_g,
              'gqq_hz': gqq_g,
              'f1_hz': f1}    
    
    # avoiding g_gc
    if show[0]=='Y':
        print('\n')
        print('guess_params')
        print(dict_g)
        print('\n')
    if show[1]=='Y':
        # select N that has best fit
        x_fc_tails = x_fc_sorted[-Ni:]
        y_tails = y_sorted[-Ni:]
        #print(len(x_fc_tails))
        #print(len(y_tails))
        Xs = x_fc_tails / (np.power(x_fc_tails, 2)-np.power(f1,2))
        #print(len(Xs))
        y_mod = slope*Xs + intercept
        plt.plot(Xs, y_tails, 'k.', label='data')
        plt.plot(Xs, y_mod, 'r-', label='guess')
        plt.xlabel(r'1/$f_c$ (GHz)')
        plt.ylabel(r'$g/2\pi$ (Hz)')
        plt.show()
    
    """Overall the guess is pretty good"""
    
    return dict_g
    
def lm_min_gqcq_hz1(xdata, ydata, gate='iswap', tol=1E4, 
                    max_amp=400E6, dict_fc={}, num_fitfrac=20, show=['Y','Y'], **kwargs):
    
    """
    Minimization routine for fitting g and g_prod_hz2 given and unknown g1c
    and g2c. We will not adjust the CZ accordingly yet. Assuming only f1
    with no detuning. This simplification makes a great minimization and guess
    for more complicated models. 
    
    Novelty on why this work is worth:
        1. Alternative way to estimate g in the advent of not measuring g_ic
        using coupler spectroscopy
        2. Good way to compare the minimization protocol with coupler swap
        to see if minimization algorithms can in the future replace 
        other protocols, minimizing reliance on extra coupling strength.
        3. Helps adjust conversion from coupling strength to negative values
        via guidance with the model.

    Parameters
    ----------
    xdata : 1D numpy array
        voltage.
    ydata : 1D numpy array
        coupling.
    guess : TYPE, optional
        DESCRIPTION. The default is [].
        
    show : TYPE, optional
        DESCRIPTION. The default is ['N','N'].
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    np.random.seed(42)
    
    """Set x-data frequency"""
    x_fc = f01_volts_to_hz(vpk=xdata, A_conv=dict_fc['A_conv'],
                           v0=dict_fc['v0'], 
                           f_max = dict_fc['f_max'], 
                           fact=dict_fc['fact'], 
                           f_Ec=dict_fc['f_Ec'])
    
    """Set-up minimization using anti-crossing functions"""
    params = Parameters()
    par = ['gic_hz', 'K', 'gqq_hz', 'f1_hz'] # all in hz
        
    # initialize parameters for simplicity of code structure
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, False, False] # default unique to the Model
        # to generalize, put things in default 
    
    dict_g = {}
    axs_lbl_fit = ['x-data', 'y-data']
    
    # second version for guess; nested if-else statement for better guess
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                # print(i)
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_g_qcq_hz1(xdata, ydata, guess[3], 
                                                     gate, dict_fc, ['N','N'],
                                                     num_fitfrac)[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2],
                      par[3]: guess[3]}
    else:
        dict_g = guess_g_qcq_hz1(xdata, ydata, guess[3], gate, dict_fc, 
                                 ['N','N'], num_fitfrac) # better at guessing parameters, works with gpg
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    # print(p)
    
    """Add Models and Evaluate"""
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], 1E5, 1E9), #gic_hz matters
                    (par[1], p[1], par_var[1], 1, 20), #K matters
                    (par[2], p[2], par_var[2], 1E5, 1E9), # need to reconvert this
                    (par[3], p[3], par_var[3]))   
    
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
        
    out = minimize(
        g_qq_hz_res1, 
        params, 
        method=method,
        scale_covar=True,
        args=(x_fc, ydata, weights, gate, tol, max_amp, 'res')
    )
    # print(out.params)
    
    """Version 2 - ordering"""
    
    # show best fit curve according to sorted branch, sort branch for visualization
    
    out_best_fit = g_qq_hz_res1(out.params, fc=x_fc, data=ydata,
                                     weights=weights, gate=gate, tol=tol, 
                                     max_amp=max_amp,
                                     output='model')
    init_fit = g_qq_hz_res1(out.params, fc=x_fc, data=ydata,
                            weights=weights, gate=gate, tol=tol, 
                            max_amp=max_amp,
                            output='model')
    
    if 'fit_res_pts' in kwargs:
        # reinitiate xdata points to varying lengths, and recalibrate
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        x_fc_eval = f01_volts_to_hz(vpk=xdata_eval, A_conv=dict_fc['A_conv'],
                                    v0=dict_fc['v0'], 
                                    f_max = dict_fc['f_max'], 
                                    fact=dict_fc['fact'], 
                                    f_Ec=dict_fc['f_Ec'])
        init_eval_fit = g_qq_hz_res1(params, fc=x_fc_eval, data=ydata,
                                         weights=weights, gate=gate, tol=tol, 
                                         max_amp=max_amp,
                                         output='model')
        out_eval_fit = g_qq_hz_res1(out.params, fc=x_fc, data=ydata,
                                         weights=weights, gate=gate, tol=tol, 
                                         max_amp=max_amp,
                                         output='model')
        # no component-fit for this fit report
        show_report_minimize(result=[out, init_eval_fit, out_best_fit], 
                             xdata=xdata, ydata=ydata, show=show, 
                             fit_eval = [xdata_eval, out_eval_fit], 
                             axis_label = axs_lbl_fit, **kwargs)
        # not sure where the params came from
        
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
    
    # edit gqq_hz according to fit.
    gic_n = out.params["gic_hz"].value
    gic_stderr = out.params["gic_hz"].stderr
    K_n = out.params["K"].value
    K_stderr = out.params["K"].stderr
    
    gic_unc = ufloat(gic_n, gic_stderr)
    K_unc = ufloat(K_n, K_stderr)
    gqq_unc = gic_unc/K_unc
    out_dict_val_stderr["gqq_hz"]=[gqq_unc.n, gqq_unc.std_dev]  
    
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

def guess_g_qcq_hz2(xdata, ydata, f1, det, gate='iswap', 
                    dict_fc={}, show=['Y','Y'], num_fitfrac = 20, **kwargs):
    """
    More comprehensive method for analyzing data. With imperfect targetting
    # f1!=f2_operation, and zz coupling !=0.
    
    Parameters
    ----------
    xdata : 1D numpy array
        voltage.
    ydata : 1D numpy array
        g-value (in hz).
    f1 : float
        operating frequency (in hz).
    det : float
        detuning between f2 and f1 in hz
    gate : TYPE, optional
        DESCRIPTION. The default is 'cz'.
    dict_fc : dictionary
        dictionary of converting biases to transmon frequencies
    num_fitfrac : int
        number of precision for linear regression fit. default = 20
    
    **kwargs : 1D numpy array
        dictionary.

    Returns
    -------
    dict_g : dictionary
        contains 'g_ic', 'g_qq', 'K'

    """
    # declaration of variables
    f2 = f1 + det
    x_fc = f01_volts_to_hz(vpk=xdata, A_conv=dict_fc['A_conv'],
                           v0=dict_fc['v0'], 
                           f_max = dict_fc['f_max'], 
                           fact=dict_fc['fact'], 
                           f_Ec=dict_fc['f_Ec'])
    
    # find residual coupling for correction. Idea, find maximum zz_g
    idx_fcm = np.where(x_fc == np.amax(x_fc))[0][0]
    zz_l = np.amin(np.abs(ydata[:idx_fcm]))
    zz_r = np.amin(np.abs(ydata[idx_fcm:]))    
    #zz_hz_g = np.where(zz_l >= zz_r, zz_l, zz_r)
    zz_hz_g = (zz_l + zz_r)/2
    
    """use correction constant for guess based on gates"""
    if gate=='iswap':
        const=1
    else:
        const=np.sqrt(2)
    
    # Sort by increasing wc for linear regression
    sorted_indices = np.argsort(x_fc)
    x_fc_sorted = x_fc[sorted_indices]
    #print(x_fc_sorted) # lowest frequency of x_fc to highest frequency of x_fc
    
    # correct magnitude of y-sorted based on quantum gate being swept
    y_sorted = ydata[sorted_indices] / const


    """
    Linear approximation: y ≈ const*(gqq - 2*gprod*X)
    where X = wc / (wc^2 - w1^2), 
    then y = m*X+b with m = -2*gprod, b=gqq,
    for better linearity, we iterate the minimum fit_fraction
    available for the guess, based on linearity rval, not 
    the most minimum g_qq.
    """
    # use fit_fractions to select region of linearity, also considers outliers.
    fit_fractions = np.linspace(0.05, 1.0, num_fitfrac)
    slopes = np.ones(num_fitfrac)
    intercepts = np.ones(num_fitfrac)
    rvals = np.ones(num_fitfrac)
    Ns = np.ones(num_fitfrac)
    for i in range(len(fit_fractions)):
        # Select high-frequency tail (wc >> w1)
        N = int(len(x_fc_sorted) * fit_fractions[i])
        x_fc_tail = x_fc_sorted[-N:]
        y_tail = y_sorted[-N:]
        
        # do linear regression of tails and extract r_val
        B = (1/(np.power(x_fc_tail, 2)-np.power(f1,2))) 
        + (1/(np.power(x_fc_tail, 2)-np.power(f2,2)))
        X = x_fc_tail*B
        slopes[i], intercepts[i], rvals[i], *_ = linregress(X, y_tail)
        rvals[i] = np.abs(rvals[i])
        Ns[i] = N
    
    # get idx of maximum rvals, pearson coefficient
    idx_lin = np.where(rvals == np.amax(rvals))[0][0]
    slope = slopes[idx_lin]
    intercept = intercepts[idx_lin]
    Ni = int(Ns[idx_lin])
        
    # guess values
    # determines gqq_g
    gqq_g = intercept - zz_hz_g
    # determine g_ic
    gprod_g = -slope
    gic_g = np.sqrt(gprod_g)
    """
    determine renormalization factor => useful as one can plug it in ranges
    from g_ic_g >> g_qq_g and one can put ranges from 1 to 20 with better
    parameter control than gic_g. Good first guess for the dataset
    """
    K_g = gic_g / gqq_g if gqq_g != 0 else 20.0  # fallback
    
    dict_g = {'gic_hz': gic_g,
              'K': K_g,
              'gqq_hz': gqq_g,
              'zz_hz': zz_hz_g,
              'f1_hz': f1,
              'det_hz': det}    
    
    # avoiding g_gc
    if show[0]=='Y':
        print('\n')
        print('guess_params')
        print(dict_g)
        print('\n')
    if show[1]=='Y':
        # select N that has best fit
        x_fc_tails = x_fc_sorted[-Ni:]
        y_tails = y_sorted[-Ni:]
        #print(len(x_fc_tails))
        #print(len(y_tails))
        B = (1/(np.power(x_fc_tails, 2)-np.power(f1,2))) 
        + (1/(np.power(x_fc_tails, 2)-np.power(f2,2)))
        Xs = x_fc_tails*B
        #print(len(Xs))
        y_mod = slope*Xs + intercept
        plt.plot(Xs, y_tails, 'k.', label='data')
        plt.plot(Xs, y_mod, 'r-', label='guess')
        plt.xlabel(r'1/$f_c$ (GHz)')
        plt.ylabel(r'$g/2\pi$ (Hz)')
        plt.show()
    
    """Check if zz_g and zz allows fit"""
    
    return dict_g

def lm_min_gqcq_hz2(xdata, ydata, gate='iswap', tol=1E4, 
                    max_amp=400E6, dict_fc={}, num_fitfrac=20, show=['Y','Y'], **kwargs):
    
    """
    Minimization routine for fitting g and g_prod_hz2 given and unknown g1c
    and g2c. We will not adjust the CZ accordingly yet. Assuming only f1
    with no detuning. This simplification makes a great minimization and guess
    for more complicated models. 
    
    Novelty on why this work is worth:
        1. Alternative way to estimate g in the advent of not measuring g_ic
        using coupler spectroscopy
        2. Good way to compare the minimization protocol with coupler swap
        to see if minimization algorithms can in the future replace 
        other protocols, minimizing reliance on extra coupling strength.
        3. Helps adjust conversion from coupling strength to negative values
        via guidance with the model.

    Parameters
    ----------
    xdata : 1D numpy array
        voltage.
    ydata : 1D numpy array
        coupling.
    guess : TYPE, optional
        DESCRIPTION. The default is [].
        
    show : TYPE, optional
        DESCRIPTION. The default is ['N','N'].
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    np.random.seed(42)
    
    """Set x-data frequency"""
    x_fc = f01_volts_to_hz(vpk=xdata, A_conv=dict_fc['A_conv'],
                           v0=dict_fc['v0'], 
                           f_max = dict_fc['f_max'], 
                           fact=dict_fc['fact'], 
                           f_Ec=dict_fc['f_Ec'])
    
    """Set-up minimization using anti-crossing functions"""
    params = Parameters()
    par = ['gic_hz', 'K', 'gqq_hz', 'f1_hz', 'det_hz', 'zz_hz'] # all in hz    
    # initialize parameters for simplicity of code structure
    
    # Check boolean for params
    if 'bool_params' in kwargs:
        par_var = kwargs['bool_params']
    else:
        par_var = [True, True, False, False, False, True] # default unique to the Model
        # to generalize, put things in default 
    
    dict_g = {}
    axs_lbl_fit = ['x-data', 'y-data']
    
    # second version for guess; nested if-else statement for better guess
    if 'guess' in kwargs:
        guess = kwargs['guess']
        if 'bool_params' in kwargs:
            # modifies guess depending on action of vary in lmfit
            for i in range(len(par_var)):
                # print(i)
                if par_var[i] == False:
                    # manually-inputting guess function depending on par var
                    dict_g[par[i]] = guess[i]
                else:
                    dict_g[par[i]] = guess_g_qcq_hz2(xdata, ydata, guess[3], 
                                                     guess[4], 
                                                     gate, dict_fc, ['N','N'],
                                                     num_fitfrac)[par[i]] # better at guessing parameters, works with gpg
        else:
            """
            guess based on intuition or on pre-fitted data, usually originating
            from published results, for testing reproducibility of model.
            """
            dict_g = {par[0]: guess[0],
                      par[1]: guess[1],
                      par[2]: guess[2],
                      par[3]: guess[3],
                      par[4]: guess[4],
                      par[5]: guess[5]}
    else:
        dict_g = guess_g_qcq_hz2(xdata, ydata, guess[3], guess[4], gate, dict_fc, 
                                 ['N','N'], num_fitfrac) # better at guessing parameters, works with gpg
    
    p = np.asarray([dict_g[elem] for elem in par]) #values of the functions
    # print(p)
    
    """Aid to guess"""
    idx_fcm = np.where(x_fc == np.amax(x_fc))[0][0]
    zz_l = np.amin(np.abs(ydata[:idx_fcm]))
    zz_r = np.amin(np.abs(ydata[idx_fcm:]))
    zz_max = np.where(zz_l>=zz_r, zz_l, zz_r)
    # print(2*zz_max)
    
    """Add Models and Evaluate"""
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)    
    
    # for ver3 => constraints dictated by key model equation 
    params.add_many((par[0], p[0], par_var[0], 1E5, 1E9), #gic_hz matters
                    (par[1], p[1], par_var[1], 1, 20), #K matters
                    (par[2], p[2], par_var[2], 1E5, 1E9), # need to reconvert this
                    (par[3], p[3], par_var[3]),
                    (par[4], p[4], par_var[4]),
                    (par[5], p[5], par_var[5], 0, 1*zz_max))   
    
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
        
    out = minimize(
        g_qq_hz_res2, 
        params, 
        method=method,
        scale_covar=True,
        args=(x_fc, ydata, weights, gate, tol, max_amp, 'res')
    )
    # print(out.params)
    
    """Version 2 - ordering"""
    
    # show best fit curve according to sorted branch, sort branch for visualization
    
    out_best_fit = g_qq_hz_res2(out.params, fc=x_fc, data=ydata,
                                     weights=weights, gate=gate, tol=tol, 
                                     max_amp=max_amp,
                                     output='model')
    init_fit = g_qq_hz_res2(out.params, fc=x_fc, data=ydata,
                            weights=weights, gate=gate, tol=tol, 
                            max_amp=max_amp,
                            output='model')
    
    if 'fit_res_pts' in kwargs:
        # reinitiate xdata points to varying lengths, and recalibrate
        xdata_eval = np.linspace(xdata[0], xdata[-1], kwargs['fit_res_pts']) #numbers spaced in logarithm
        x_fc_eval = f01_volts_to_hz(vpk=xdata_eval, A_conv=dict_fc['A_conv'],
                                    v0=dict_fc['v0'], 
                                    f_max = dict_fc['f_max'], 
                                    fact=dict_fc['fact'], 
                                    f_Ec=dict_fc['f_Ec'])
        init_eval_fit = g_qq_hz_res1(params, fc=x_fc_eval, data=ydata,
                                         weights=weights, gate=gate, tol=tol, 
                                         max_amp=max_amp,
                                         output='model')
        out_eval_fit = g_qq_hz_res1(out.params, fc=x_fc, data=ydata,
                                         weights=weights, gate=gate, tol=tol, 
                                         max_amp=max_amp,
                                         output='model')
        # no component-fit for this fit report
        show_report_minimize(result=[out, init_eval_fit, out_best_fit], 
                             xdata=xdata, ydata=ydata, show=show, 
                             fit_eval = [xdata_eval, out_eval_fit], 
                             axis_label = axs_lbl_fit, **kwargs)
        # not sure where the params came from
        
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
    
    # edit gqq_hz according to fit.
    gic_n = out.params["gic_hz"].value
    gic_stderr = out.params["gic_hz"].stderr
    K_n = out.params["K"].value
    K_stderr = out.params["K"].stderr
    # print(gic_n)
    gic_unc = ufloat(gic_n, gic_stderr)
    K_unc = ufloat(K_n, K_stderr)
    gqq_unc = gic_unc/K_unc
    out_dict_val_stderr["gqq_hz"]=[gqq_unc.n, gqq_unc.std_dev]  
    
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