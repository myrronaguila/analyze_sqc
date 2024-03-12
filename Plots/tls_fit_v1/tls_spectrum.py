# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:04:26 2024

TLS spectrum and analysis - from Klimov et al 2018
To deploy as version 1

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time # benchmarking time
import random

# import lmfit as lm
# import scipy as sp

import uncertainties.unumpy as unumpy 
from uncertainties import ufloat

#for KDE
from scipy import stats
from scipy.io import savemat, loadmat
from sklearn.neighbors import KernelDensity
from lmfit.models import GaussianModel, ConstantModel, LorentzianModel #for pdf, selecting threshold
from sklearn.model_selection import GridSearchCV

#for median over absolute deviations and median of a distribution
#from scipy import stats

# data plotting
# from data_plots_qm import cm_to_inch
# from asqum_format import find_nearest, lst_xyrange #for data conditioning
from matplotlib.ticker import AutoMinorLocator

"""----------------------Array search and filter----------------------------"""
def find_nearest(array, value):
    """Find index and element to an array to which its value is nearest to the
    reference value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def check_and_flatten(obj):
    if isinstance(obj, list):
        if all(isinstance(item, list) for item in obj):
            # Object is a list of multiple lists, flatten it
            return flatten_list(obj)
        elif all(isinstance(item, float) for item in obj):
            # Object is a list of floats
            return obj
    # return None

"""----------------------Plot formatting------------------------------------"""
def cm_to_inch(x):
    # formatting data plots from image from cm to inches
    return x/2.54

def assign_rand_color(num_col):
    """
    Assign random color based on assigned numbers

    Parameters
    ----------
    num_col : float
        number of needed colors.

    Returns
    -------
    color_arr . list of strings
        colors in hexadecimal structure
    """
    # get number of random colors from hexadecimal pairs
    hexadecimal_alphabets = '0123456789ABCDEF'
    color_arr = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in 
                                range(6)]) for i in range(num_col)]
    return color_arr

def color_num_pairs(ax, x_bnds, y_bnds, string, c_arr):
    """
    Color distinct vlines or h-lines according to set range.
    Function determines automatically if h-lines or v-lines are needed depending
    on the nature of the bounds at the expense of computational time. 
    Sets faint background in between for better contrast between list assignments
    
    - works only if the x_bnds and y_bnds are > 2, else we use ax.vlines or
    ax.hlines explicitly
    
    Parameters
    ----------
    ax : axes
        axis whose background to be drawn
    x_bnds : list of listed pairs / or list of pairs
        x-axis pairs that need distinct color / can also act as x-bounds if pair='2'
    y_bnds : list of pairs / or list of listed pairs
        y-axis pairs that need distinct color / can also act as y-bounds if pair='2'.
    string : string
        'x' => pair of xlines
        'y' => pair of vlines
    c_arr : list of colors per pairs
        color pairs of pair of lines for frequency match
    Returns
    -------
    None.

    """
    # number pair minimum is pairs - makes bounds
    num_x_pairs = len(check_and_flatten(x_bnds))//2 # floor division 
    num_y_pairs = len(check_and_flatten(y_bnds))//2 # floor division
    
    # define routines for plotting ax.x-axis
    def color_line_pairs(ax, string, x_bnds, y_bnds, c_arr):
        """
        number of v-line pairs

        Parameters
        ----------
        ax : axes
            list of strings
        string : string
            'y' - y axis
            'x' - x axis
        x_bnds : 1D numpy array
            return
        y_bnds : list of pair of floats
            sets y-min and y-max on plots            

        Returns
        -------
        None.

        """
        
        # draw dashed lines
        if string == 'y':
            # set random colors
            num_pairs = len(x_bnds)
            color_arr = assign_rand_color(num_pairs)
            # draw num_pairs of pair of vertical lines
            [ax.vlines(x = x_bnds[i], ymin = y_bnds[0], ymax=y_bnds[1], 
                       ls='dashed', color = color_arr[i]) for i in range(num_pairs)]
            # y-bnds is constant
            [ax.fill_between(x=x_bnds[i], y1=y_bnds[0], y2=y_bnds[1], facecolor =c_arr[i], 
                             alpha = 0.5) for i in range(num_pairs)]
        else: 
            # set random colors
            num_pairs = len(y_bnds)
            color_arr = assign_rand_color(num_pairs)
            # draw num_pairs of pair of horizontal lines
            [ax.hlines(y = y_bnds[i], xmin = x_bnds[0], xmax=x_bnds[1], 
                       ls='dashed', color = color_arr[i]) for i in range(num_pairs)]
            # x-bnds is constant
            [ax.fill_between(x=x_bnds, y1=y_bnds[i][0], y2=y_bnds[i][1], facecolor =c_arr[i], 
                             alpha = 0.5) for i in range(num_pairs)]
    
    # set conditions for coloring ranges    
    if int(num_x_pairs) > int(num_y_pairs):
        # y-line pairs as bounds, num_x_pairs as pair of vertical lines
        color_line_pairs(ax, 'y', x_bnds, y_bnds, c_arr)
    elif int(num_x_pairs) < int(num_y_pairs):
        # x-line pairs as bounds, num_y_pairs as pair of horizontal lines
        color_line_pairs(ax, 'x', x_bnds, y_bnds, c_arr)
    else:
        color_line_pairs(ax, string, x_bnds, y_bnds, c_arr)
        
"""----------------------Peak filter algorithm------------------------------"""
def median_threshold_algo(x, y, xlag=[], xmerge=[], threshold=1, num_pts= 7, show='N'):
    """
    Simple static threshold based on median filter of measured algorithms. 
    No scored Z-counts. Without assumption of gaussian normal distribution, 
    the median value will be the baseline. Data set serves to provide
    semi-automatic information of the guess peaks for TLS spectrum. One can
    refine the guess by omitting artifacts in frequency, and data.
    
    
    # if median is only considered, this will result to noisy data. But if we
    consider the permissivible standard deviation of the whle algorithm, then
    some outliers can be ignored.
    
    The median_threshold_algo helps in obtaining guess functions for the plots,
    Parameters
    ----------
    y : 1D numpy array
        Y-data to quantify.
    xlag : list of tuple or list
        list of frequency range to be omitted in the analysis
        lag[i] = [freq_start_i, freq_end_i]
        # lag[i] = [index_start_i, index_end_i]
    xmerge : list of tuple or list
        list of frequency range to be merged in the signals
    threshold : float
        factor of the median of average deviation
    num_pts : float
        number of points near the peak frequency (must be odd)
    show : string
        'Y' = show test plot
        
    Returns
    -------
    dict(signals = np.array(signals),
         medFilter = np.array(med)
         madFilter = np.array(mad))
    dict_guess_sp : dictionary of guesses from points
    """
    
    """Retrieve signals from semi-automatic filters"""
    
    signals = np.zeros(len(y))
    y_med = np.median(y)
    medFilter = y_med*np.ones(len(y)) #assuming non-updating distribution
    madFilter = stats.median_abs_deviation(y) #getting standard deviation of filter
    #y_med -= 2*madFilter # somewhat lower so that it is buried in noise
    
    # determine range of 1s and zeros to connote peak widths
    for i in range(len(y)):
        if abs(y[i] - medFilter[i]) > threshold*madFilter:
            if y[i] > medFilter[i]:
                signals[i] = 1
            else:
                signals[i] = -1
        else:
            signals[i] = 0
    
    # remove signals with user-known false positives
    if len(xlag) > 0:
        for i in range(len(xlag)):
            """Get minimum and maximum range possible"""
            idx_a = find_nearest(array=x, value=xlag[i][0])[0]
            idx_b = find_nearest(array=x, value=xlag[i][1])[0]
            signals[idx_a:idx_b]=0
            
    # connect signals with adjacent 0 and 1s for a certain lorentzian bandwidth
    if len(xmerge) > 0:
        for i in range(len(xmerge)):
            """Get minimum and maximum range possible"""
            idx_a = find_nearest(array=x, value=xmerge[i][0])[0]
            idx_b = find_nearest(array=x, value=xmerge[i][1])[0]
            signals[idx_a:idx_b]=1
    
    """
    Group signals that have ones and get their indices to obtain maximum,
    sigma and height per peak as guess
    """
    
    """Find indices of elements with value 1"""
    indices_of_ones = np.where(signals == 1)[0]

    # Group consecutive indices
    groups = []
    temp_group = [indices_of_ones[0]]
    for i in range(1, len(indices_of_ones)):
        if indices_of_ones[i] == indices_of_ones[i - 1] + 1:
            temp_group.append(indices_of_ones[i])
        else:
            groups.append(temp_group)
            temp_group = [indices_of_ones[i]]

    # Append the last group
    groups.append(temp_group)
    # print(groups)
    
    # find indices for 0 and -1 for getting the median and median standard deviation
    flattened_indices = np.concatenate(groups)
    ungrouped = np.setdiff1d(np.arange(len(signals)), flattened_indices)
    # print(ungrouped)
    
    # updated filter after removing peaks that are considered as outliers for base peak
    y_med2 = np.median(y[ungrouped])
    medFilter2 = y_med2*np.ones(len(y))
    y_mad2 = stats.median_abs_deviation(y[ungrouped])
    
    # retrieve list of arrays for groups
    x_gp = [x[group] for group in groups]
    y_gp = [y[group] for group in groups]    
    n_gp = len(groups) # number of identified peaks, 8
    
    # grab non-one groups to determine new median and median std
    
    """
    Find indices of elements with value 0 and -1 for obtaining the new median
    
    """
    
    """ retrieve parameters for n-identified peaks"""
    
    # identify peak frequency and points near the identified peaks
    num_points = num_pts # points near the peaks
    # even if num_points exceed length of an array, it will be limited by the points
    
    #peak_idx = np.argmax(y_gp[0])
    idx_pk_gp = [np.argmax(y_gp[i]) for i in range(len(y_gp))]
    
    # x_fpk_gp = [x_gp[i][np.argmax[y_gp[i]]] for i in range(len(x_gp))]
    start_idx_gp = [max(0, idx_pk_gp[i] - num_points // 2) for i in range(n_gp)]
    end_idx_gp = [min(len(x), idx_pk_gp[i] + num_points // 2 + 1) for i in range(n_gp)]
    
    # guess peak-bounds and linewidth => f_pk_bnds and 2*sigma
    sel_x_gp = [x_gp[i][start_idx_gp[i]:end_idx_gp[i]] for i in range(n_gp)] 
    sel_y_gp = [y_gp[i][start_idx_gp[i]:end_idx_gp[i]] for i in range(n_gp)]
    
    # guess peak frequency, upper and lower bounds
    fpk_gp = [x_gp[i][idx_pk_gp[i]] for i in range(len(x_gp))]
    fpk_gp_bnds = [(sel_x_gp[i][0] ,sel_x_gp[i][-1]) for i in range(n_gp)]
        
    # guess peak linewidth = sigma = fwhm/2
    sigma_gp = [np.abs(sel_x_gp[i][-1] - sel_x_gp[i][0])/2 for i in range(n_gp)] #from points
    sigma_gp_bnds = [(0, np.abs(x_gp[i][0]-x_gp[i][-1])/2) for i in range(n_gp)]
    
    # guess peak amplitude using the updated median
    # height in lorentzian = (1/np.pi)*(amplitude/sigma)
    # amp = height*sigma*np.pi
    #apk_gp = [(y_gp[i][idx_pk_gp[i]]-y_med)*np.pi*sigma_gp[i] for i in range(n_gp)]
    
    # guess peak amplitude using the updated median (without the potential peaks)
    apk_gp = [(y_gp[i][idx_pk_gp[i]]-y_med2)*np.pi*sigma_gp[i] for i in range(n_gp)]
    apk_gp_bnds = [(0, 2*apk_gp[i]) for i in range(n_gp)]
        
    """make dictionary of guess for the lorentzian as output"""
    
    dict_guess_gp = {'pk_freq': fpk_gp,
                     'pk_freq_bnds': fpk_gp_bnds,
                     'sigma': sigma_gp,
                     'sigma_bnds': sigma_gp_bnds,
                     'pk_amp': apk_gp,
                     'pk_amp_bnds': apk_gp_bnds,
                     'gam_qbt': y_med2,
                     'gam_qbt_bnds': (0, y_med2 + 2*y_mad2)}
            
    # show plots for cross-check
    if show=='Y': 
        
        wfig=8.6
        fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                           1*cm_to_inch(wfig)))
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace =0.1, wspace=0.1)
        ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model
        ax1 = fig.add_subplot(spec[1, 0]) # line-width plot

        # set color of pair of ranges assigned by xmerge
        if len(xmerge)>0: 
            color_arr = assign_rand_color(len(xmerge))

        #Klimov S2
        ax0.plot(x, y, 'k-', label='Data')
        ax0.plot(x, medFilter, 'c--', label='Median')
        ax0.plot(x, medFilter2, 'g:', label='New Median')
        [ax0.plot(sel_x_gp[i], sel_y_gp[i], 'b.') for i in range(n_gp)]
        ax0.set_yscale('log')
        ax0.set_xlabel(r'$f_q$ (GHz)')
        ax0.set_ylabel('1/T$_1$')
        ax0.set_xlim(x[0], x[-1])
        # ax0.set_ylim(2E-2, 7)
        # set xmerge range for clarity
        # print(ax0.get_ylim())
        # print([ax0.get_ylim()[0],ax0.get_ylim()[1]])
        if len(xmerge)>0: 
            color_num_pairs(ax=ax0, x_bnds=xmerge, 
                            y_bnds=[ax0.get_ylim()[0],ax0.get_ylim()[1]], 
                            string='y', c_arr = color_arr)
        ax0.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                        right = True)
        ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax0.legend(loc='best', frameon=False)

        # Klimov Signals
        ax1.plot(x, signals, color="red", lw=2, ls=None, marker='.', ms=1, label='med')
        ax1.set_xlabel(r'$f_q$ (GHz)')
        ax1.set_ylabel('Signal')
        ax1.set_xlim(x[0], x[-1])
        # xet xmerge range for clarity
        # print(ax1.get_ylim())
        # print([ax1.get_ylim()[0],ax1.get_ylim()[1]])
        if len(xmerge) > 0: 
            color_num_pairs(ax=ax1, x_bnds=xmerge, 
                            y_bnds=[ax1.get_ylim()[0],ax1.get_ylim()[1]], 
                            string='y', c_arr = color_arr)
        ax1.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                        right = True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax1.legend(loc='best', frameon=False)
        plt.show()
    
    return dict(signals = np.asarray(signals),
                medFilter = np.asarray(medFilter),
                madFilter = np.asarray(madFilter)), dict_guess_gp

"""---------------------Model TLS Lorentzian Spectrum-----------------------"""
"""-----------------data retrieval from dictionaries------------------------"""
def bundle_params_from_dict(dict_nlz, param_lst = [['hwhm_q'], ['g_h','f_tls', 
                                                              'hwhm_tls']]):
    """
    Group parameters for dictionary for iterative curve-fit from multi-lorentzian
    fit for quick analysis [useful for building scatter plots with errors]
    
    Dictionary outputs numpy arrays for iterables and floats for non-iterables
    # example of iterables are lz1... lzn
    # example of non-iterables are 
    
    Default params contains keywords from lz1_ ... lzn_ except hwhm_q
    But these parameters can be applied
    
    #note. Haven't tested'
    
    Bundling is useful in plotting fit lorentzians to know the scatter of
    extracted parameters (See Klimov et al, S2b)
    
    Parameters
    ----------
    dict_nlz : dictionary of lists
        dictionary of data (default is TLS dataset that has the following
                            datastrings)
    param_lst : list of various parameters
        param_lst[0] = list of string
            list of string that are have single-values throughout the fit 
        param_lst[1] = list of strings
            list of strings that are iterable by numerous lz (creates 1D numpy
                                                              arrays)
    
    Note: parameters represent number of Lorentzian and non-lorentzian peaks
    
    Returns
    -------
    dict_reform : dictionary
        Examples of keys based on inputs are the following
        keys = ['lz_lst', 'hwhm_q_n', 'hwhm_q_std', 'g_h_n', 'g_h_std', 
                'f_tls_n', 'f_tls_std', 'hwhm_tls_n', 'hwhm_std'] => key model for TLS
        or
        keys = ['lz_lst', 'const_c_n', 'const_c_std', 'amplitude_n', 'amplitude_std', 
                'center_n', 'center_std', 'sigma_n', 'sigma_std'] => from Lorentzian model
    """
    # count the number of lorentzians or params by selecting one parameter, #chatgpt solution
    suffix = param_lst[1][0]
    count = sum(1 for key in dict_nlz if key.endswith(suffix))  
    lz_string = 'nlz'
    lz_list = ['lz%d' % (i+1) for i in range(count)]
    
    #add _n and _std in the labels for the new dictionary keys, chatgpt solutions
    #flat_list = [item for sublist in param_lst for item in (sublist if isinstance(sublist, list) else [sublist])]
    flat_list = [item for sublist in param_lst for item in sublist]
    key_list = [item + '_n' for item in flat_list] + [item + '_std' for item in flat_list]
    # print(key_list)
    # ['hwhm_q_n', 'g_h_n', 'f_tls_n', 'hwhm_tls_n', 'hwhm_q_std', 'g_h_std', 'f_tls_std', 'hwhm_tls_std']
    # print('\n')
    
    # bundle 1D numpy arrays from each parameters
    # sequence of keys depend on sequence of strings set in param_lst[1] = ['g_h', 'f_tls', 'hwhm_tls']
    values_n = [np.asarray([value[0] for key, value in dict_nlz.items() if key.endswith(flat_list[i])]) 
             for i in range(len(flat_list))]
    values_std = [np.asarray([value[1] for key, value in dict_nlz.items() if key.endswith(flat_list[i])]) 
             for i in range(len(flat_list))]
    values_lst = values_n + values_std
    # print(values_lst)
    # print('\n')
    
    # construct dict for the parameters
    dict_iter = {key_list[i]: values_lst[i] for i in range(len(key_list))}
    # add dict-keys for 'nlz' with lz_list last 
    dict_iter[lz_string] = lz_list
    
    return dict_iter

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

""" ------ LMFit using N-Lorentzian functions with constant background------"""
def add_peak(prefix, center, sigma=0.05, amplitude=0.005):
    # from internet, to be modified
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[prefix + 'center'].set(center)
    pars[prefix + 'amplitude'].set(amplitude)
    pars[prefix + 'sigma'].set(sigma, min=0)
    return peak, pars

def add_peak_mod(prefix, center, sigma, amplitude, bnds, par_vary=[True, True, True]):
    """
    modified Lorentzian fit format with informed based on semi-automatic guess
    
    Refer to example 3 fitting multiple peaks and with
    bounds  https://lmfit.github.io/lmfit-py/builtin_models.html        
    
    Priority is all 8 peaks have good fitting with literature.
    
    Parameters
    ----------
    prefix : string
        label of Lorentzian.
    center : float
        peak frequency.
    amplitude : float
        peak amplitude.
    sigma : float
        peak half-width half max.
    bnds : list of tuples of lower and upper bounds
        bnds[0] = (cen_lb, cen_up).
        bnds[1] = (sigma_lb, sigma_up)
        bnds[2] = (amp_lb, amp_up).
    par_vary : list of strings
        # refer to set on https://lmfit.github.io/lmfit-py/parameters.html
        par_vary[0] = 'Vary' or None => for center
        par_vary[1] = 'Vary' or None => for sigma
        par_vary[2] = 'Vary' or None => for amplitude
    
    Returns
    -------
    peak : lm funct
        Modelled Lorentzian peaks.
    pars : lm funct
        parameters.
    """
      
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    
    #trial 2: constant peak frequency by toggling center vary to false
    pars[prefix + 'center'].set(value=center, vary=par_vary[0], min=bnds[0][0],
                                max=bnds[0][1])
    #comment: good location with the expense of bad peak amplitude on peak 1
    
    pars[prefix + 'sigma'].set(value=sigma, vary=par_vary[1], min=bnds[2][0], 
                               max=bnds[2][1])
    
    pars[prefix + 'amplitude'].set(value=amplitude, vary=par_vary[2], min=bnds[1][0], 
                                   max=bnds[1][1])
    
    return peak, pars

def lm_nlz_const_fit(xdata, ydata, guess_dict, par_vary=[True, True, True, True],
                     show=['Y','Y'], bundle='N', save=['N', 'file'], **kwargs):
    """
    Multi-lorentzian fit with constant background using lmfit.
    # reference = https://stackoverflow.com/questions/57278821/how-does-one-fit-multiple-independent-and-overlapping-lorentzian-peaks-in-a-set 
    
    Parameters
    ----------
    xdata : 1D npy array
        Frequency (GHz).
    ydata : 1D np array
        1/T1 (MHz)
    guess : dictionary of values based on thresholding algorithm
        dict_guess_gp = {'pk_freq': list of frequency peaks,
                         'pk_freq_bnds': list of tuple of frequency bounds,
                         'sigma': list of linewidths based on threshold,
                         'sigma_bnds': list of tuple of lower and upper lw bnds,
                         'pk_amp': list of lorentzian amplitudes,
                         'pk_amp_bnds': list of amplitude bounds,
                         'gam_qbt': value,
                         'gam_qbt_bnds': (0, y_med + 2*madFilter)}
    par_vary : list of booleans
        par_vary[0] = True or False (boolean, on const)
        par_vary[1] = True or False (boolean, on pk freq)
        par_vary[2] = True or False (boolean, on pk sigma)
        par_vary[3] = True or False (boolean, on pk amplitude)
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    bundle : string
        'Y' => make all Lorentzian fit parameters listed in 1 array
        'N' => standard lmfit format
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
    **kwargs : key-ordered argument in dictionary
        time : string 
            'Y' or 'N' => display time or not.
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing
                sweep_info[0] : string
                    'main_param_name [i.e. time_min, power_W, etc]'
                sweep_info[1] : float
                    parameter value
                sweep_info[2] : list
                    comments on other parameters ['string', float]
                    default is []
                    similar to param_name and sweep-info but made easier to classify
                    in future analysis
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_vales:  dictionary of results
        best fit parameters
    """
    
    # start time
    start_time = time.time()
    
    # build list of guess functions from guess_dict
    pk_freqs = guess_dict['pk_freq']
    pk_freqs_bnds = guess_dict['pk_freq_bnds']
    sigmas = guess_dict['sigma']
    sigmas_bnds = guess_dict['sigma_bnds']
    pks_amp = guess_dict['pk_amp']
    pks_amp_bnds = guess_dict['pk_amp_bnds']
    bkg_qbt = guess_dict['gam_qbt'] #Qubit Half-width Half maximum => linewidth by barends
    bkg_qbt_bnds = guess_dict['gam_qbt_bnds']
    bkg_qbt_std = (bkg_qbt_bnds[1] - bkg_qbt)/2 #what is obtained here is MAD
        
    """Set-up model"""
    const_mod = ConstantModel(prefix='const_')
    # ['const_c']
    
    # first option
    # params = const_mod.guess(ydata, x=xdata)
    # no good because it over-estimated the peaks
    
    params = const_mod.make_params(c=dict(value=bkg_qbt, vary=par_vary[0], 
                                          min=bkg_qbt_bnds[0], 
                                          max=bkg_qbt_bnds[1]))
    # provided an improvement in the fit
    par_list = ['const_c']
    
    
    for i, cen in enumerate(pk_freqs):
       
        # trial 4 function, only frequency is fixed
        pref = 'lz%d_' % (i+1)
        peak, pars = add_peak_mod(prefix=pref, center=cen, 
                                  sigma=sigmas[i], amplitude=pks_amp[i],
                                  bnds=[pk_freqs_bnds[i], pks_amp_bnds[i], 
                                        sigmas_bnds[i]], par_vary=par_vary[1:])
        # comment. no change in amplitude unless we extend the fitting range of lorentziants
        # otherwise there is improvements
        const_mod += peak
        params.update(pars)
        
        # add independent labels to parameters
        lbl_params = ['amplitude', 'center', 'sigma', 'fwhm', 'height']
        [par_list.append(pref + lbl) for lbl in lbl_params] #list comprehension
        
    init = const_mod.eval(params, x=xdata)
    out = const_mod.fit(ydata, params, x=xdata)
    comps = out.eval_components()
    
    """Report Results - quite verbose, need a function to reduce words"""
    if show[0] == 'Y':
        print('\n')
        #this report is important as it gives measure on further analysis the option of bundling
        print(out.fit_report(min_correl=0.25))
        print('\n')
        
    if show[1] == 'Y':
        wfig=8.6
        fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                           2*cm_to_inch(wfig)))
        spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace =0.1, wspace=0.1)
        ax0 = fig.add_subplot(spec[0, 0]) # data, initial and final fit
        ax1 = fig.add_subplot(spec[1, 0]) # data and fit breakdown

        # Compare between initial and best fit
        ax0.plot(xdata, ydata, 'k.', label='Data')
        ax0.plot(xdata, init, 'b:', label='Init')
        ax0.plot(xdata, out.best_fit, 'r-', label='Best Fit')
        ax0.set_yscale('log')
        ax0.set_xlabel(r'$f_q$ (GHz)')
        ax0.set_ylabel('1/T$_1$')
        ax0.set_xlim(xdata[0], xdata[-1])
        ax0.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                        right = True)
        ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax0.legend(loc='best', frameon=False)

        # Breakdown in fit components
        ax1.plot(xdata, ydata, 'k.', label='Data')
        ax1.plot(xdata, out.best_fit, 'r-', label='Best Fit')
        for name, comp in comps.items():
            ax1.plot(xdata, comp, ':', label=name)
        ax1.set_yscale('log')
        ax1.set_xlabel(r'$f_q$ (GHz)')
        ax1.set_ylabel('1/T$_1$')
        ax1.set_xlim(xdata[0], xdata[-1])
        ax1.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                        right = True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax1.legend(loc='best', frameon=False, ncols=4, fontsize='small')
        plt.show()
    
    """For multi-peak analysis better reorganize the dictionary according to
    the following: gam_qubit, gam_qubit_err, lz1, lz2, ..., lz(n-1), lzn"""
    print('\n')
    # print(out.var_names) => output parameters involved in the variation of parameters
    # par_names must output its own list of names independently
    # print(out.params) output a special parameter class
    # print(params) #output same as print_params
    # print(par_list) # par_list works
    
    """Save results"""
    # this commands only work if par_vary[:]='True'
    
    # iterate best fit based on the vary conditions
    out_dict_val_stderr = {par_list[i]: [out.params[par_list[i]].value, 
                                         out.params[par_list[i]].stderr]
                            for i in range(len(par_list))}
    
    """
    for const_c, there is already a guess algorithm, and an mad can be included 
    For amplitude, center and sigma, no need to update the standard variation.
    to note the user about the information in the dictionary, simply append a key and labels
    """
    if par_vary[0] == False:
        out_dict_val_stderr['const_c'][1] = bkg_qbt_std
    
    # query if data bundling is allowed
    if bundle == 'Y':
        out_dict_val_stderr = bundle_params_from_dict(dict_nlz=out_dict_val_stderr,
                                                      param_lst=[['const_c'],
                                                                 ['amplitude', 'center', 'sigma', 'fwhm', 'height']])
    
    #add fit conditions to inform the user on fit condition for reproducibility
    out_dict_val_stderr['vary_info'] = ['const_c', 'lz_center', 'lz_sigma', 'lz_amplitude']
    out_dict_val_stderr['vary_bool'] = par_vary
    
    # add other data parameters for future analysis
    if 'input_info' in kwargs:
        out_dict_val_stderr['input_info'] = kwargs['input_info']
    
    # save dictionary for data references and comparison
    if save[0] == 'Y':
        if len(save) == 1:
            # use default name to save data
            savemat('n_lor+const_bkg.mat', out_dict_val_stderr)
        else: 
            savemat(save[1] + '.mat', out_dict_val_stderr)
        
        
    # record time elapsed
    end_time = time.time()
    execution_time = round(end_time - start_time,2)
    
    if kwargs['time']=='Y':
        print('\n')
        print(f'Elapsed time = {execution_time} seconds')
           
    return out.best_fit, out_dict_val_stderr
    
def map_nlz_to_tls(dict_nlz):
    """
    Convert Lorentzian parameters in dictionary to dictionary of relevant
    TLS parameters - see the library 

    Parameters
    ----------
    dict_nlz : dictionary
        Dictionary from nlz.

    Returns
    -------
    float.
    """
    # set initial name for dictionary
    lz_params = ['amplitude', 'center', 'sigma', 'fwhm', 'height']
    str_params = ['const_c', 'vary_bool', 'vary_info']
    # note that Gamma_i is already expressed as gamma_i/2pi = Hz
    tls_params = ['g_h','f_tls', 'hwhm_tls', 'fwhm_tls', 'depth_tls']
    
    # count the number of lorentzians in the dictionary
    n_lz = int((len(dict_nlz) - len(str_params))/len(lz_params))
    # print(n_lz)
    
    # copy whole dictionary in python 
    tls_dict_val_stderr = dict_nlz.copy() 
    # this preserves fit params from nlz
    
    """
    copy hwhm_q, f_tls, hwhm_tls, fwhm_tls and depth_tls, only modify g_h
    rename const_c with hwhm_q => qubit params.
    """
    
    tls_dict_val_stderr['hwhm_q'] = tls_dict_val_stderr.pop('const_c')
    
    def g_h(amp_lst):
        """
        Convert amplitude to g/h (Hz or MHz depending on the initial scale)

        Parameters
        ----------
        amp_lst : list
                [amp.n, amp.stderr]

        Returns
        -------
        list of floats.
            [g_h.n, g_h.std]
        """
        amp_u = ufloat(amp_lst[0], amp_lst[1])
        pi = ufloat(np.pi, 0.0)
        val = unumpy.sqrt(amp_u/(2*pi))
        return [float(unumpy.nominal_values(val)), float(unumpy.std_devs(val))]    
    
    # use for loops to rename all components and edit the components
    
    # for loop version
    for i in range(n_lz):
        pref = 'lz%d_' % (i+1)
        # rename dict tls
        for j in range(len(lz_params)):
            lz_lbl = pref + lz_params[j]
            tls_lbl = pref + tls_params[j] 
            tls_dict_val_stderr[tls_lbl] = tls_dict_val_stderr.pop(lz_lbl)
            if tls_params[j] == 'g_h':
                tls_dict_val_stderr[tls_lbl] = g_h(dict_nlz[lz_lbl])
    
    return tls_dict_val_stderr

def lm_nlz_tls_fit(xdata, ydata, guess_dict, par_vary=[True, True, True, True], 
                   show=['Y','Y'], bundle='Y', save=['N', 'file'], **kwargs):
    
    """                  
    lm_nlz fit but modified the library to get pertinent characteristics of
    the tls spectrum

    Parameters
    ----------
    xdata : 1D npy array
        Frequency (GHz). x-axis ()
    ydata : 1D np array
        1/T1 (MHz) y-axis (default)
    guess : dictionary of values based on thresholding algorithm
        dict_guess_gp = {'pk_freq': list of frequency peaks,
                         'pk_freq_bnds': list of tuple of frequency bounds,
                         'sigma': list of linewidths based on threshold,
                         'sigma_bnds': list of tuple of lower and upper lw bnds,
                         'pk_amp': list of lorentzian amplitudes,
                         'pk_amp_bnds': list of amplitude bounds,
                         'gam_qbt': value,
                         'gam_qbt_bnds': (0, y_med + 2*madFilter)}
    par_vary : list of booleans
        par_vary[0] = True or False (boolean, on const)
        par_vary[1] = True or False (boolean, on pk freq)
        par_vary[2] = True or False (boolean, on pk sigma)
        par_vary[3] = True or False (boolean, on pk amplitude)
    show : list of strings, optional
        show[0] = show report
        show[1] = show best of fit. The default is ['Y','Y'].
    bundle : string
        'Y' => make all Lorentzian fit parameters listed in 1 array
        'N' => standard lmfit format
    save : list of string
        save[0] = 'N' or 'Y'
        save[1] = 'filename' if save[0]= 'Y'
        save[2] = 'N' or 'Y' => Enable data bundling (iterate all LZ in one plot)
    **kwargs : key-ordered argument in dictionary
        time : string 
            'Y' or 'N' => display time or not.
        input_info : list
            Labeling measurement parameters of nLor-fit for further processing.
            For ease in readability, just make the list an iterable.
                input_info[0] : string
                    param 1 name [i.e. time_min, power_W, etc]'
                input_info[1] : float
                    param 1 value
                input_info[2] : string
                    param 2 name [i.e. time_min, power_W, etc]'
                input_info[3] : float
                    param 2 value
                ... etc.
    Returns
    -------
    best fit values. 1D numpy array
        best fit
    out_dict_vales:  dictionary of results
        best fit parameters
    """

    # start time
    start_time = time.time()

    # call multiple lorentzian with baseline function
    best_fit, out_dict = lm_nlz_const_fit(xdata=xdata, ydata=ydata, 
                                          guess_dict=guess_dict, par_vary=par_vary, 
                                          show=['Y','Y'], bundle='N', save=['N', 'file'], 
                                          time='N')
    
    # map out lorentzian fit results to interacting TLS model
    tls_dict = map_nlz_to_tls(dict_nlz=out_dict)
    # Bundle key params in another dictionary for defect lorentzian
    tls_specs = bundle_params_from_dict(dict_nlz=tls_dict, 
                                        param_lst=[['hwhm_q'], 
                                                   ['g_h','f_tls', 'hwhm_tls', 'fwhm_tls', 'depth_tls']])
    #add fit conditions to inform the user on fit condition for reproducibility
    tls_specs['vary_info'] = ['const_c', 'lz_center', 'lz_sigma', 'lz_amplitude']
    tls_specs['vary_bool'] = par_vary
    
    # replot lorentzian based on tls model
    ntls_mod = nlz_tls(xdata=xdata, fi_arr=tls_specs['f_tls_n'], 
                       hwhm_tls_arr=tls_specs['hwhm_tls_n'], 
                       g_tls_arr=tls_specs['g_h_n'], 
                       hwhm_q=tls_specs['hwhm_q_n'])
    
    """show function is verbose, need to set better plots for this"""
    
    if show[1]=='Y':
        # check tls model and raw data
        wfig=8.6
        fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                           1*cm_to_inch(wfig)))
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
        ax0 = fig.add_subplot(spec[0, 0]) # Example of Lorentzian Fit
        ax1 = fig.add_subplot(spec[0, 1])

        ax0.set_title('Example Relaxation Resonance Fit')
        ax0.plot(xdata, ydata, 'r.', label='Data')
        ax0.plot(xdata, best_fit, 'b-', label='MultiLor Fit')
        ax0.plot(xdata, ntls_mod, 'g--', label='TLS model')
        ax0.set_yscale('log')
        ax0.set_xlabel(r'$f_q$ (GHz)')
        ax0.set_ylabel('1/T$_1$ (MHz)')
        ax0.set_xlim(xdata[0], xdata[-1])
        ax0.tick_params(axis = 'both', which ='both', direction='in', top=True, right = True)
        ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
        ax0.legend(loc='best', frameon=False)

        """Conversion from hwhm to Gamma_i from Klimov"""
        e1 = 1 #converting to MHz
        e2 = 1*1E3 #converting to MHz
        eq = 1 
        """plot"""

        ax1.set_title('Defect Lorentzian Best fit Parameters')
        ax1.errorbar(x=tls_specs['g_h_n']*e1, y=tls_specs['hwhm_tls_n']*e2, 
                     yerr=tls_specs['hwhm_tls_std']*e2, xerr=tls_specs['g_h_std']*e1,
                     fmt='.', ecolor='k', color='k', markersize=8, label='TLS')
        ax1.hlines(y=tls_specs['hwhm_q_n']*eq, xmin=ax1.get_xlim()[0], 
                   xmax=ax1.get_xlim()[1], color='blue', label='Qubit')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('$g/h$ (MHz)')
        ax1.set_ylabel(r'HWHM (MHz)')
        ax1.tick_params(axis = 'both', which ='both', direction='in', top=True, right = True)
        ax1.legend(loc='best', frameon=False)
        plt.show()
    
    # query if retrieving bundle data or standard lmfit data format
    if bundle=='Y':
        tls_dict_save = tls_specs
    else:
        tls_dict_save = tls_dict
    
    # save option is verbose... better make scalable codes.
    
    if 'input_info' in kwargs:
        tls_dict_save['input_info'] = kwargs['input_info']
    
    if save[0] == 'Y':
        if len(save) == 1:
            # use default name to save data
            savemat('tls+const_qbt.mat', tls_dict_save)
        else: 
            # save[1] stands  for filename but maintaining lmfit format
            savemat(save[1] + '.mat', tls_dict_save)
    
    # record time elapsed
    end_time = time.time()
    execution_time = round(end_time - start_time,2)
    if kwargs['time']=='Y':
        print('\n')
        print(f'Elapsed time = {execution_time} seconds')
    
    return best_fit, tls_dict_save

"""----------------------Parameter Retrieval From .mat----------------------"""
def loadmat_to_dict_1d(fname):
    """
    Retrieve dictionary from loadmat without changing the dictionary format
    for data retrieval. Note that when dictionary is produced by the software,
    the arrays are created as lists. But by saving the dataset in .mat file,
    the lists becomes 2D numpy arrays with the shape [1,n] where n is the length
    of a list. If the list contains boolean, .mat convert it to 1 (True)
    or 0 (False). If the list contains an array of floats, .mat converts it
    to arrays of 1D numpy array. List of strings are converted to numpy modules
    
    We retain as much info as possible. The loadfile is not applicable for
    intentional dictionaries with 2D arrays created.
    
    Parameters
    ----------
    fname : string
        filenames.

    Returns
    -------
    dictionary.
    """
    dict_1 = loadmat(fname)
    
    for key, value in dict_1.items():
        # flatten np.arrays with [1,n] to 1D numpy arrays with length n
        if len(dict_1[key]) == 1:
            dict_1[key] = dict_1[key].flatten()
            # convert input_info from 1D numpy array to list
            
        if key == 'input_info':
            param_o = dict_1[key].tolist()
            # convert even elements to float values, while removing whitespace
            param_o = [float(param_o[i]) if i%2 ==1 else str(param_o[i]).strip() 
                        for i in range(len(param_o))]
            dict_1[key] = param_o
        
        # convert boolean numbers to True or False
        if key == 'vary_bool':
            converted_values = [True if num == 1 else False for num in dict_1[key]]
            dict_1[key] = converted_values
        
        # convert str numpy to list of strings
        if key == 'nlz':
            dict_1[key] = dict_1[key].tolist()
            # remove trailing and following whitespace
            dict_1[key] = [dict_1[key][i].strip() for i in range(len(dict_1[key]))]    
        
        # convert str numpy to list of strings
        if key == 'vary_info':
            dict_1[key] = dict_1[key].tolist()
            # remove trailing and following whitespace
            dict_1[key] = [dict_1[key][i].strip() for i in range(len(dict_1[key]))]
    
    # formatting is finished.
    return dict_1

"""---------Distribution Function For Checking Baseline time----------------"""

def ecdf(a):
    #ecdf depending on array of data
    #https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python
    # a represents set of data obtained from the distribution over long time spans
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

def get_ecdf(a):
    """
    Get an empirical cumulative distribution function from the arrays of data
    obtained from the results

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    """
    #convert ecdf to x,y cdf for step-post
    x, y = ecdf(a)
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    #plt.plot(x, y, drawstyle='steps-post')
    return x,y

def get_epdf(a):
    """
    Get an empirical probability distribution from the arrays of data obtained 
    from the results

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    x : TYPE
        DESCRIPTION.
    y_pdf : TYPE
        DESCRIPTION.

    """
    #extract epdf from step-post, query from chatgpt-misses the mark
    x, y = ecdf(a)
    y_pdf = np.diff(y)
    #x = np.insert(x, 0, x[0]) #initiate line
    y_pdf = np.insert(y_pdf, 0, 0.) #probability distribution starts from 0.
    return x, y_pdf

def npdf_to_cdf(y_pdf):
    """Convert numerical pdf to cdf"""
    #obtain numerical cumulative distribution function from KDE
    y_cdf_opt = np.cumsum(y_pdf)
    y_cdf_opt = y_cdf_opt / np.max(y_cdf_opt)
    return y_cdf_opt

"""Can be remodelled as a class of functions"""


"""---------Fitting Gaussian Model------------------------------------------"""
def Gaussian(x, mu, sigma, Amp):
    # Set an adjustable gaussian distribution
    # for a normalized Gaussian Distribution, Amp=1
    A = 1/(sigma*np.sqrt(2*np.pi))
    B = np.exp(-(x-mu)**2/(2*(sigma**2)))
    return Amp*A*B

def lm_gauss_fit(xdata, ydata, show=['N','N']):
    """
    Gaussian fit from PDF to obtain a statistic using LMFIT library
    Input :
      xdata : 1D numpy array
        x-data fit
      ydata : 1D numpy array
        y-data fit
      show : list of strings
        show[0] = 'Y' - show fit report
        show[1] = 'Y' - show fitting
    """
    mod = GaussianModel()
    pars = mod.guess(ydata, x=xdata)
    init = mod.eval(pars, x=xdata)
    out = mod.fit(ydata, pars, x=xdata)
    """--Retrieve extracted data from fitting function-------"""
    name = ['amplitude', 'center', 'sigma','fwhm', 'height']
    # strange dictionary
    out_dict_val_stderr = {name[i]: [out.params[name[i]].value,
                                    out.params[name[i]].stderr] for i in range(len(name))}
    if show[0] == 'Y':
        print(out.fit_report(min_correl=0.25))
    if show[1] == 'Y':
        #show amplitude fit
        wfig = 8.6
        fig = plt.figure(constrained_layout=True, figsize=(cm_to_inch(wfig),
                                                           cm_to_inch(wfig)))
        spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.05, wspace=0.1)
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(xdata, ydata)
        ax1.plot(xdata, init, '--', label='initial fit')
        ax1.plot(xdata, out.best_fit, '-', label='best fit')
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('PDF')
        ax1.legend()

    
    return out.best_fit, out_dict_val_stderr

"""------------------------optimized KDE------------------------------------"""
#Create optimizer for empirical probability density distribution
def optimize_kde_pdf(xdata, ydata, bwid_lst, show='Y'):
    """
    Optimize Kernel Density Distribution of Stochastic Data
    
    core issue with this type of optimization is that we could have better
    information on what to minimize if we obtain a minimum function or
    interpolate parameters minimizing gridsearch
    
    Input:
      xdata : 1D numpy array
        interpolating data
      ydata: 1D numpy array
        y-data values to be trained
      kde_bwidth_lst : list of values
        kde_bwid_lst[0] : starting bandwidth based on sample data
        kde_bwid_lst[1] : ending bandwidth based on sample data
        kde_bwid_lst[2] : step bwidth based on sample data
    return:
      y_pdf_opt: 1D numpy array
        Optimized Probability Density Function
      y_cdf_opt: 1D numpy array
        Corresponding Data Functional Distribution
    """
    # prepare necessary arrays for kde analysis
    xdata_interp = xdata[:, np.newaxis]
    ydata_train = ydata[:, np.newaxis]
    h_vals = np.arange(bwid_lst[0], bwid_lst[1], bwid_lst[2])
    kernels = ['cosine', 'epanechnikov', 'exponential', 'gaussian',
               'linear', 'tophat']
    #prepare scoring for kernels (data filters at certain bandwidths)
    def my_scores(estimator, X):
        """
        Giving scores to a filter function
        Input :
          estimator : string
            Kernels
          X : 1d numpy array
            float
        """
        scores = estimator.score_samples(X)
        # Remove -inf
        scores = scores[scores != float('-inf')]
        # Return the mean values
        return np.mean(scores)

    #Perform Gridsearch for KDE
    grid = GridSearchCV(KernelDensity(),
                      {'bandwidth': h_vals, 'kernel': kernels},
                      scoring=my_scores)
    # grid-seach is limited by the resolution of h_vals
    grid.fit(ydata_train)
    best_kde = grid.best_estimator_
    log_dens = best_kde.score_samples(xdata_interp) #log max likelihood ratio

    #show report for fit
    if show == 'Y':
      print("Best Kernel: " + best_kde.kernel+" h="+"{:.2f}".format(best_kde.bandwidth))

    #Calculate for Probability Density estimation
    y_pdf_opt = np.exp(log_dens)  #maximum likelihood expressed in PDF

    #obtain numerical cumulative distribution function from KDE
    y_cdf_opt = np.cumsum(y_pdf_opt)
    y_cdf_opt = y_cdf_opt / np.max(y_cdf_opt)
    return y_pdf_opt
