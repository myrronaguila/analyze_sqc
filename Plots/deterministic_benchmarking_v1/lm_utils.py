# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:55:36 2024

Utilities to improve fitting via lmfit. Concerns of lmfit data formatting,
retrieval, data-shaping, saving and loading, and fit reports. Goal is to make
codes readable especially for plotting figures.

@author: Mai
"""

import numpy as np
import scipy as sp
import peakutils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from scipy.io import loadmat, savemat
from data_plots_qm import cm_to_inch, line_plot

import time as time

#20241025
from lmfit import report_fit
"""---------Getting Benchmark times for certain functions-------------------"""
# make subroutine for decorators
def time_perf(func):
    def wrapper(*args, **kwargs):
        # for precision and monotonic small codes, time.perf_counter is good for short codes
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.3e} seconds')
        return result
    return wrapper

def time_time(func):
    def wrapper(*args, **kwargs):
        # for long-piece of codes, time.time() is good benchmark
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print(f'Time taken: {elapsed:.3e} seconds')
        return result
    return wrapper

"""----------Retrieve Data for Analysis-------------------------------------"""

def get_1D_plot_from_csv(filename):
    """
    Get Data from filenames (2 columns)

    Parameters
    ----------
    filename : string
        1D numpy array.

    Returns
    -------
    xdata : 1D numpy array
        DESCRIPTION.
    ydata : TYPE
        DESCRIPTION.

    """
    df = pd.DataFrame(pd.read_csv(filename + '.csv', header=None))
    xdata = df[0].to_numpy()
    ydata = df[1].to_numpy()
    return [xdata, ydata]

def retrieve_data_from_csv_list(directory, fname_list):
    """
    Fetch list of files based on pandas csv. Subroutine simplifies retrieval of 
    x,y data according to fname_list order. 

    Parameters
    ----------
    directory : string
        folder name and directory
    fname_list : list of strings
        filenames under the folder without .csv.

    Returns
    -------
    dictionary of data:
        {idx_fname_list[i]: [x[i],y[i]]}
    """
    # initialize dictionary
    dict_data = {}
    
    # append data to dictionary according to naming order
    for i in range(len(fname_list)):
        dict_data[i] = get_1D_plot_from_csv(directory + fname_list[i])    
    return dict_data

def dict_to_array_lists(dic):
    """
    Convert dictionary to lists of arrays

    Parameters
    ----------
    dic : dictionary
        dictionary of data.
        keys = [0, 1, 2, etc.] => ideas
        

    Returns
    -------
    x2Ddata : list of 1D array
        data
    y2Ddata : list of 1D array
        data
    """
    x2Ddata = [dic[key][0] for key in sorted(dic.keys())]
    y2Ddata = [dic[key][1] for key in sorted(dic.keys())]
    return x2Ddata, y2Ddata

def retrieve_2D_lists_from_csv_list(directory, fname_list):
    dict_data = retrieve_data_from_csv_list(directory, fname_list)
    x2Ddata, y2Ddata = dict_to_array_lists(dic=dict_data)
    return x2Ddata, y2Ddata 

def get_data_from_mat(filename):
    """
    Retrieve Data from .mat file

    Parameters
    ----------
    filename : string
        filename of data.

    Returns
    -------
    dict: dictionary

    """
    df = loadmat(filename)
    return df

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
            dict_1[key] = dict_1[key].flatten() #maintain 1D numpy array
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

def get_IQ_from_csv(filename):
    """
    Retrieve Frequency, I and Q Data from .csv file

    Parameters
    ----------
    filename : string
        filename of data.

    Returns
    -------
    freq: 1D numpy array
        frequency in GHz
    dict: dictionary of arrays
        dict['I'] = 1D array In-quadrature
        dict['Q'] = 1D array Out-of-plane quadrature
        dict['Amp'] = 1D array Amplitude
        dict['Phase'] = 1D array Phase
    """
    df = pd.DataFrame(pd.read_csv(filename + '.csv'))
    freq_GHz_arr = df['<b>frequency(GHz)</b>'].values

    I = df['I'].values #in V_ratio
    Q = df['Q'].values #in V_ratio
    Amp = np.sqrt(I**2 + Q**2) #in V_ratio
    #Phase = np.unwrap(p=np.arctan(Q/I), period=np.pi) #in radians, python 3.11
    Phase = np.unwrap(p=np.arctan(Q/I)) #in radians, python 3.11
    

    """Pack data in dictionary"""
    dict_val = {'freq_GHz': freq_GHz_arr,
                'I': I, 
                'Q': Q,
                'Amp': Amp,
                'Phase': Phase}
    return freq_GHz_arr, dict_val

def get_IQ_from_csv_2(filename, xstring):
    """
    updated: 20230530 - Retrieve Frequency, I and Q Data from .csv file

    Parameters
    ----------
    filename : string
        filename of data.
    xstring : string
        can be frequency='<b>frequency(GHz)</b>' in GHz or 
        time domain='<b>relaxation_time</b>' in ns

    Returns
    -------

    dict: dictionary of arrays
        dict['I'] = 1D array In-quadrature
        dict['Q'] = 1D array Out-of-plane quadrature
        dict['Amp'] = 1D array Amplitude
        dict['Phase'] = 1D array Phase
    """
    df = pd.DataFrame(pd.read_csv(filename + '.csv'))
    x_data = df[xstring].values
    
    if xstring == '<b>frequency(GHz)</b>':
        xlabel = 'freq_GHz'
    elif xstring == '<b>relaxation_time</b>':
        xlabel = 'relax_time_ns'
    elif xstring == '<b>T2</b>':
        xlabel = 'dephasing_time_ns'
    elif xstring == '<b>T1</b>':
        xlabel = 'relax_time_ns'

    I = df['I'].values #in V_ratio
    Q = df['Q'].values #in V_ratio
    Amp = np.sqrt(I**2 + Q**2) #in V_ratio
    #Phase = np.unwrap(p=np.arctan(Q/I), period=np.pi) #in radians, python 3.11
    Phase = np.unwrap(p=np.arctan(Q/I)) #in radians, python 3.11
    
    """Pack data in dictionary"""
    dict_val = {xlabel: x_data,
                'I': I, 
                'Q': Q,
                'Amp': Amp,
                'Phase': Phase}
    return dict_val

def freq_range(x_data, x_a, x_b):
    """Equivalent of Numpy as making range in frequencies (MHz)"""
    idx_a = np.where(x_data >= x_a)[0][0] # getting the first element of min data
    idx_b = np.where(x_data <= x_b)[0][-1] # getting the last element of max data
    return x_data[idx_a:(idx_b+1)], idx_a, int(idx_b+1)

def lst_xyrange(lst_xdata, lst_ydata, x_a, x_b):
    """
    Extract a smaller list of 1D arrays of data

    Parameters
    ----------
    lst_xdata : list of 1D arrays
        list of x-data.
    lst_ydata : list of 1D arrays
        list of y-data.
    x_a : float
        lower frequency.
    x_b : float
        higher frequency.

    Returns
    -------
    Data Analysis.
    """
    lst_xdat = []
    lst_ydat = []
    n = len(lst_xdata)
    for i in range(n):
        """Get minimum and maximum range possible"""
        idx_a = np.where(lst_xdata[i] >= x_a)[0][0]
        idx_b = np.where(lst_xdata[i] <= x_b)[0][-1]
        # print(idx_a)
        # print(idx_b)
        """Append Empty array"""
        lst_xdat.append(lst_xdata[i][idx_a:idx_b])
        lst_ydat.append(lst_ydata[i][idx_a:idx_b])
    return lst_xdat, lst_ydat

def find_nearest(array, value):
    """Find index and element to an array to which its value is nearest to the
    reference value"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def get_mins_from_2Darray(z2D, x, par_arr, min):
  """
  Extract peak parameters per parameter in 2D array
  Input :
    z2D : 2D numpy array
      amplitude in 2D array:
      2D numpy array shape = (len(par_arr), len(x))
    x : 1D numpy array
      frequency spectrum
    par_arr : 1D numpy array
      varied parameters for analysis
    min : string
      'dip' => minimum
      'peak' => maximum
  """
  n = len(par_arr)
  par_extract = np.ones(n)
  for i in range(len(par_arr)):
    if min == 'dip':
      idx=find_nearest(array=z2D[i], value=np.amin(z2D[i]))[0]
    elif min == 'peak':
      idx=find_nearest(array=z2D[i], value=np.amax(z2D[i]))[0]
    else:
      idx=find_nearest(array=z2D[i], value=np.amax(z2D[i]))[0]
    par_extract[i] = x[idx]

    #check for accuracy
    #print(np.amax(z2D[i]))
    #print(idx)
    #print(x[idx])
  return par_extract

def detect_mins(x_arr, y_arr, string, args):
    """
    get peaks and dips above a threshold amplitude for a 1D-array using peak
    utils. Useful for getting peaks in X and Y axis. Very susceptible to noise
    unless filter functions are implemented.

    Parameters
    ----------
    x_arr : 1D numpy array
        1D data array.
    y_arr : 1D numpy array
        1D data array.
    string : 1D numpy array
        string=peaks.
        string=dips.
    args : 1D numpy array
        args[0] = threshold amplitude
        args[1] = minimum spacing
        args[2] = number of max peaks

    Returns
    -------
    min_list : list of list of values
        min_list[0] = list of indices
        min_list[1] = list of x-value floats
        min_list[2] = list of y-value floats
    """
    if string == 'peaks':
        indices = peakutils.indexes(y_arr, thres=args[0], min_dist=args[1])
        pass
    elif string=='dips':
        y_arr*= -1
        indices = peakutils.indexes(y_arr, thres=np.abs(args[0]), min_dist=args[1])
    else:
        indices = []
    return indices

def parse_z_to_comps(z):
    """
    Express complex data to I, Q, Amp and Phase
    
    Parameters
    ----------
    z : 1D numpy array
        complex-valued data.
    
    Returns
    -------
    i : 1D numpy array
        Real component / in-phase
    q : 1D numpy array
        Imaginary component / out-of-phase
    amp : 1D numpy array
        Amplitude
    phase : 1D numpy array
        Phase in radians
    """
    i = np.real(z)
    q = np.imag(z)
    amp = np.abs(z)
    phase = np.angle(z)
    return i, q, amp, phase

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
    # can be considered as ctrl_dic
    suffix = param_lst[1][0]
    count = sum(1 for key in dict_nlz if key.endswith(suffix))  
    lz_string = 'nlz'
    lz_list = ['lz%d' % (i+1) for i in range(count)]
    # useful for multipeak data in one plot
    
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
    
    #data_dict and ctrl_dic similar to this data framework, 
    #the codes lack units, best put in include in input_info
    return dict_iter

def info_retriv(fitpars, string):
    """
    retrieve tuple of fit parameters (nominal values and errors) from fit
    
    For memory efficiency, slicing the variable would be better rather than
    two list comprehension
    
    20240627 - better edit this list_comprehension so that err_1D can appear as 
    a linear format rather than a tuple
    
    Parameters
    ----------
    fitpars : list of dictionaries
        format = [{}]. fitpars is an array of dictionaries of values
    string : string
        retrieved filename of the dictionaries

    Returns
    -------
    val_1D : 1D numpy array
        Nominal Values of parameters.
    err_1D : 1D numpy array
        Stderr of parameters.

    """
    val_1D = np.asarray([fitpars[i][string][0] for i in range(len(fitpars))])
    err_1D = np.asarray([fitpars[i][string][1] for i in range(len(fitpars))])
    """Output are values of both errors and 1D string"""
    return val_1D, err_1D

def dict_retriv_form(data_dic, ctrl_dic, save):
    """
    Repackage list of dictionaries to saveable dictionary

    Parameters
    ----------
    dic : list of dictionaries containing tuple of fits
        Data from Dictionary after autofit.
        dic = [{var1: [var1.n, var1.std], var2: [var2.n, var2.std]},
               {var1: [var1.n, var1.std], var2: [var2.n, var2.std]},]
    var_dic : Variable Dic
        Dictionary with variables
    save : list of strings
        save[0]='Y': save data
        save[0]='N': do not save
        save[1]='filename of .mat'

    Returns
    -------
    None.

    """

    #acquire the list of names for a dictionary
    names = list(data_dic[0].keys()) #take first dictionary containing keys
    # for ease in labeling units, embed info of names and put them as {list_vars: names_units}
    
    #add control list in front of the list
    ctrl_str = list(ctrl_dic.keys())[0] #only takes first column, containing independent variable
    
    # print(var_str)
    # print(list(ctrl_dic.keys()))
    # print(var_str)
    names.insert(0, ctrl_str)
    
    #create a new dictionary of parameters
    dic_new = {names[i]: ctrl_dic[ctrl_str] if names[i]==ctrl_str else
               info_retriv(fitpars=data_dic, string=names[i]) for i in range(len(names))}
    
    #label lies in info_retriv from names
    #second row create fitpars

    #note that dictionary produced creates a tuple of data. Best to classify

    #save dictionary if necessary
    if save[0]=='Y':
        savemat(file_name = save[1]+'.mat',mdict=dic_new)

    return dic_new

def dict_retriv_rev(data_dic, ctrl_dic, save):
    """
    Simplified dictionary retrieve 20240713

    Parameters
    ----------
    data_dic : TYPE
        DESCRIPTION.
    ctrl_dic : TYPE
        DESCRIPTION.
    save : TYPE
        DESCRIPTION.

    Returns
    -------
    ctrl_dic : TYPE
        DESCRIPTION.

    """
    
    #acquire list of names for a dictionary with relabels
    names_dat = list(data_dic[0].keys())
    
    # retrieve data from tuples and sort them into dictionary keys
    dic_data = {name: info_retriv(fitpars=data_dic, string=name) for name in names_dat}
    
    # Initialize an empty dictionary for the sorted data
    sorted_dic_data = {}
    
    # Iterate over the original dictionary
    for key, value_tuple in dic_data.items():
        for i, array in enumerate(value_tuple):
            if i == 0:
                new_key = f'{key}_n'
            else:
                new_key = f'{key}_err'
            sorted_dic_data[new_key] = array
    
    # insert ctrl_dic on top of data for Qerr
    ctrl_dic.update(sorted_dic_data)

    if save[0]=='Y':
        savemat(file_name = save[1]+'.mat',mdict=ctrl_dic)
    
    return ctrl_dic
    

"""
dict_retriv_form, bundle_params_from_dict and info_retriv can be compressed in 
one comprehensive retrieval docs idea is to solve the data-format analysis in 
one swoop for easy plotting and retrieval need to revise bundle-params 2.0 for ubiquitous processes
"""

def conc_val_er(ar1, ar2, er1, er2):
    #Reference: https://stackoverflow.com/questions/1720421/how-do-i-concatenate-two-lists-in-python
    return [*ar1, *ar2], [*er1, *er2]

def change_key_and_value_in_dict(d, old_keys, new_keys, new_values):
    """
    overwrite existing keys and values in dictionary. useful for
    non-rewriting fits if the fitting algorithm is good enough
    
    Parameters
    ----------
    d : dictionary
        dictionary of fit parameter nominal and fit values.
    old_keys : list of strings
        old keys from old dictionaries.
    new_keys : list of strings
        new keys to replace from dictionaries.
    new_values : list of values / 1D arrays
        new values / arrays that would be evaluated

    Returns
    -------
    dictionary.

    """
    for i in range(len(old_keys)):
        if old_keys[i] in d:
            del d[old_keys[i]]          # Remove the old key-value pair
            # first trial, works but the action do not rearrange the dictionary
            #d[new_keys[i]] = new_values[i]  # Assign the new value to the new key
            
            # second trial, retains the dictionary released by the old system 
            items = list(d.items()) # make d into list of tuples
            items.insert(i, (new_keys[i], new_values[i])) #insert new pairs
            d = dict(items) # convert rearranged data to lists
        else:
            print(f"Key '{old_keys[i]}' not found in dictionary.")
    return d

"""-----------------Data Plotting and Reporting from LMFIT Functions--------"""
def fine_tune_fit():
    """perform brute optimization for better fitting candidates
    we tried log, weights etc but the fit could not converge to theoretical.
    We wanted error-bars on all data. We think minimizer is better than
    model, and more flexible but perhaps we consider using the brute method
    for finding minimum. But this requires additional preknowledge of the
    device, hence we need additional information to control the bounds of
    the fit. I think the brute optimization looks promising but only after
    considering the theoretical foundation of the device.
    """
    
    return

def show_report(result, xdata, ydata, show=['N', 'N'], **kwargs):
    """
    Show resulting fit report from objective lmfits only - using models

    Parameters
    ----------
    result : lmfit parameters
        resulting lmfit parameters after fitting - from lmfit.Model.
    xdata : 1D numpy array
        data in float.
    ydata : TYPE
        DESCRIPTION.
    show : TYPE, optional
        DESCRIPTION. The default is ['N', 'N'].
        first string refers to fit report
        second string refers to plot fit vs data
    **kwargs : dictionary and keys
        axis_label = ['x-data', 'y-data'] #useful for test
        axis_title = 'var_amp=1 units; plot i/n' # food for noting files
        axis_annotate = 'const_vars0 = var0; const_vars1 = var1' #note database,
        can be added as next line of axis_title but with smaller font
        fit_res_pts = [1001] 
        axis_scale = ['log', 'linear'] => [xscale, yscale]
        
    Returns
    -------
    None.

    """
    # retrieve inputs    
    init = result.init_fit
    
    #use kwargs for improved resolution
    if 'fit_eval' in kwargs:
        # command to automatically increase data-points for evaluation of modelled fit
        xdata_eval = kwargs['fit_eval'][0]
        #overwrite out and comps for evaluation
        out = kwargs['fit_eval'][1]
        comps = kwargs['fit_eval'][2] 
        # this option is best added externally because the models are out
    else:
        # default evaluation
        out = result.best_fit
        comps = result.eval_components()
        
    #set set figure size
    wfig=8.6
    
    if show[0] == 'Y':
        print('\n')
        #this report is important as it gives measure on further analysis the option of bundling
        print(result.fit_report(min_correl=0.10))
        print('\n')
    
    if show[1] == 'Y':
        
        """Set global parameters for plot report before checking complex data"""
        #color plots for data, init and best fit
        col_fit = [['k', '.', 'solid'], ['b', None, 'dotted'], ['r', None, 'solid']]
        
        #we do not care much about ticks2D as of the moment
        ticks_null = [[],[]]
        
        #format plots using input values in kwargs
        if 'axis_label' in kwargs:
            axs_lbl = kwargs['axis_label'] # which is a list of x and y labels
        else: 
            axs_lbl = ['x-data', 'I (V)', 'Q (V)', 'Amp (V)', 'Phase (Rad)']
        font_size = 9
        
        if 'axis_title' in kwargs:
            axs_tit = kwargs['axis_title']
        else:
            axs_tit = ''
        
        # print(np.iscomplexobj(ydata)) #just to check true or false, should be false
        if np.iscomplexobj(ydata):
            """
            if ydata is complex-valued, show six plots: 
                I,Q,Amp,Phase vs x-axis, (4 plots)
                IQ plane, (1 plot) and  
                I vs x-axis with component fits (6 plots)
            """
            #arrange datasets for complex-valued ydata, init and out
            i_data, q_data, amp_data, phase_data = parse_z_to_comps(ydata)
            i_init, q_init, amp_init, phase_init = parse_z_to_comps(init)
            
            # make if and else statement for quality fit - overwrite out
            i_best, q_best, amp_best, phase_best = parse_z_to_comps(out)
            
            #bundle related data for IQ vs x
            i_dat = [i_data, i_init, i_best]
            q_dat = [q_data, q_init, q_best]
            amp_dat = [amp_data, amp_init, amp_best]
            phase_dat = [phase_data, phase_init, phase_best]
            
            if 'fit_eval' in kwargs:
                x_dat = [xdata if i <= 1 else xdata_eval for i in range(len(i_dat))]
            else:
                x_dat = [xdata for i in range(len(i_dat))]
            
            name, comp = zip(*comps.items())
            yi_comp = list(np.real(comp))
            lbl_comp = list(name)
            
            if 'fit_eval' in kwargs:
                x_dat_comp = [xdata_eval for i in range(len(lbl_comp))]
            else:
                x_dat_comp = [xdata for i in range(len(lbl_comp))]
            
            #end of retrieving possible info from kwargs
            
            # label raw data
            lbl_I = ['raw I', 'init-fit I', 'best-fit I']
            lbl_Q = ['raw Q', 'init-fit Q', 'best-fit Q']
            lbl_amp = ['raw Amp', 'init-fit Amp', 'best-fit Amp']
            lbl_phase = ['raw Phase', 'init-fit Phase', 'best-fit Amp']
            lbl_IQ = ['raw IQ', 'init-fit IQ', 'best_fit IQ']
            lbl_comp0 = ['raw I', 'best-fit I']
            
            # comps items will be generated from fit
            
            #plot figures X and Y data
            fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                               2*cm_to_inch(wfig)))
            spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, hspace =0.1, wspace=0.1)
            ax0 = fig.add_subplot(spec[0, 0]) # I data vs fit
            ax1 = fig.add_subplot(spec[0, 1]) # Q data vs fit
            ax2 = fig.add_subplot(spec[1, 0]) # Amp data vs fit
            ax3 = fig.add_subplot(spec[1, 1]) # Phase data vs fit
            ax4 = fig.add_subplot(spec[0, 2]) # I vs Q fit
            ax5 = fig.add_subplot(spec[1, 2]) # I breakdown + components
            
            # format for line_plot 
            # line_plot(ax, x_dat, y_dat, axs_lbl, lbl_arr, col_mark, ticks_2D, font_size)
            
            # create best-fit for first 4 plots
            # put title to only ax0
            ax0.set_title(axs_tit)
            line_plot(ax0, x_dat, i_dat, axs_lbl[:2], lbl_I, col_fit, 
                      ticks_null, font_size)
            ax0.legend(loc='best', frameon=False)
            
            line_plot(ax1, x_dat, q_dat, [axs_lbl[0], axs_lbl[2]], lbl_Q, 
                      col_fit, ticks_null, font_size)
            ax1.legend(loc='best', frameon=False)
            
            line_plot(ax2, x_dat, amp_dat, [axs_lbl[0], axs_lbl[3]], lbl_amp, 
                      col_fit, ticks_null, font_size)
            ax2.legend(loc='best', frameon=False)
            
            line_plot(ax3, x_dat, phase_dat, [axs_lbl[0], axs_lbl[4]], lbl_phase, 
                      col_fit, ticks_null, font_size)
            ax3.legend(loc='best', frameon=False)
            
            line_plot(ax4, i_dat, q_dat, [axs_lbl[1], axs_lbl[2]], lbl_IQ, 
                      col_fit, ticks_null, font_size)
            ax4.legend(loc='best', frameon=False)
            
            # breakdown in comps plot - otherwise rewrite line_plots for readability
            line_plot(ax5, x_dat[:2], [i_dat[0], i_dat[2]], axs_lbl[:2], 
                      lbl_comp0, [col_fit[0], col_fit[2]], 
                      ticks_null, font_size)
            # line_plot(ax, x_dat, y_dat, axs_lbl, lbl_arr, col_mark, ticks_2D, font_size)
            line_plot(ax5, x_dat_comp, yi_comp, axs_lbl[:2], 
                      lbl_comp, [], ticks_null, font_size)
            ax5.legend(loc='best', frameon=False, fontsize='small')
            plt.show()
            
        else: 
            
            # list parameters to be plotted
            y_dat = [ydata, init, out]
            
            if 'fit_eval' in kwargs:
                #only out fit is modified
                x_dat = [xdata if i <= 1 else xdata_eval for i in range(len(y_dat))]
            else:
                x_dat = [xdata for i in range(len(y_dat))]
            
            name, comp = zip(*comps.items())
            y_comp = list(comp)
            lbl_comp = list(name)
            if 'fit_eval' in kwargs:
                x_dat_comp = [xdata_eval for i in range(len(lbl_comp))]
            else: 
                x_dat_comp = [xdata for i in range(len(lbl_comp))]
            
            # create parameters for fit
            lbl_fit = ['raw', 'init-fit', 'best-fit']
            lbl_comp0 = ['raw', 'best-fit']
            
            # formulate data
            fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                               2*cm_to_inch(wfig)))
            spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace =0.1, wspace=0.1)
            
            ax0 = fig.add_subplot(spec[0, 0]) # data, initial and final fit
            ax1 = fig.add_subplot(spec[1, 0]) # data and fit breakdown
            
            # plot 1
            ax0.set_title(axs_tit)
            line_plot(ax0, x_dat, y_dat, axs_lbl[:2], lbl_fit, col_fit, 
                      ticks_null, font_size)
            ax0.legend(loc='best', frameon=False)
            
            # breakdown of fits
            # different marker
            line_plot(ax1, [x_dat[0],x_dat[2]], [y_dat[0],y_dat[2]], 
                      axs_lbl[:2], lbl_comp0, col_fit, ticks_null, 
                      font_size)
            # same marker
            line_plot(ax1, x_dat_comp, y_comp, axs_lbl[:2], 
                      lbl_comp, [], ticks_null, font_size)
            ax1.legend(loc='best', frameon=False, ncols=4, fontsize='small')
            
            if 'axis_scale' in kwargs:
                ax0.set_xscale(kwargs['axis_scale'][0])
                ax0.set_yscale(kwargs['axis_scale'][1])
                
                ax1.set_xscale(kwargs['axis_scale'][0])
                ax1.set_yscale(kwargs['axis_scale'][1])
            plt.show()
    return

def show_report_minimize(result, xdata, ydata, show=['N', 'N'], **kwargs):
    """
    Show resulting fit report from objective lmfits only - using lmfit.minimize
    Remove components

    Parameters
    ----------
    result : lmfit parameters
        resulting lmfit parameters after fitting - from lmfit.minimize.
        result = [residual_init, residual_best_fit]
    xdata : 1D numpy array
        data in float.
    ydata : TYPE
        DESCRIPTION.
    show : TYPE, optional
        DESCRIPTION. The default is ['N', 'N'].
        first string refers to fit report
        second string refers to plot fit vs data
    **kwargs : dictionary and keys
        axis_label = ['x-data', 'y-data'] #useful for test
        axis_title = 'var_amp=1 units; plot i/n' # food for noting files
        axis_annotate = 'const_vars0 = var0; const_vars1 = var1' #note database,
        can be added as next line of axis_title but with smaller font
        fit_res_pts = [1001] 
        axis_scale = ['log', 'linear'] => [xscale, yscale]
        
    Returns
    -------
    None.

    """
    res = result[0]
    init = result[1]
    best_fit = result[2]
    
    #use kwargs for improved resolution
    if 'fit_eval' in kwargs:
        # command to automatically increase data-points for evaluation of modelled fit
        xdata_eval = kwargs['fit_eval'][0]
        #overwrite out and comps for evaluation
        out = kwargs['fit_eval'][1]
        #comps = kwargs['fit_eval'][2] 
        # this option is best added externally because the models are out
    else:
        # default evaluation
        out = best_fit
        #comps = result.eval_components()
    
    #set set figure size
    wfig=8.6
    
    if show[0] == 'Y':
        print('\n')
        #this report is important as it gives measure on further analysis the option of bundling
        #print(result.fit_report(min_correl=0.10))
        print(report_fit(res, min_correl=0.1))
        print('\n')
    
    if show[1] == 'Y':
        
        """Set global parameters for plot report before checking complex data"""
        #color plots for data, init and best fit
        col_fit = [['k', '.', 'solid'], ['b', None, 'dotted'], ['r', None, 'solid']]
        
        #we do not care much about ticks2D as of the moment
        ticks_null = [[],[]]
        
        #format plots using input values in kwargs
        if 'axis_label' in kwargs:
            axs_lbl = kwargs['axis_label'] # which is a list of x and y labels
        else: 
            axs_lbl = ['x-data', 'I (V)', 'Q (V)', 'Amp (V)', 'Phase (Rad)']
        font_size = 9
        
        if 'axis_title' in kwargs:
            axs_tit = kwargs['axis_title']
        else:
            axs_tit = ''
        
        # print(np.iscomplexobj(ydata)) #just to check true or false, should be false
        if np.iscomplexobj(ydata):
            """
            if ydata is complex-valued, show six plots: 
                I,Q,Amp,Phase vs x-axis, (4 plots)
                IQ plane, (1 plot) and  
                I vs x-axis with component fits (6 plots)
            """
            #arrange datasets for complex-valued ydata, init and out
            i_data, q_data, amp_data, phase_data = parse_z_to_comps(ydata)
            i_init, q_init, amp_init, phase_init = parse_z_to_comps(init)
            
            # make if and else statement for quality fit - overwrite out
            i_best, q_best, amp_best, phase_best = parse_z_to_comps(out)
            
            #bundle related data for IQ vs x
            i_dat = [i_data, i_init, i_best]
            q_dat = [q_data, q_init, q_best]
            amp_dat = [amp_data, amp_init, amp_best]
            phase_dat = [phase_data, phase_init, phase_best]
            
            if 'fit_eval' in kwargs:
                x_dat = [xdata if i <= 1 else xdata_eval for i in range(len(i_dat))]
            else:
                x_dat = [xdata for i in range(len(i_dat))]
            
            #name, comp = zip(*comps.items())
            #yi_comp = list(np.real(comp))
            #lbl_comp = list(name)
            
            #if 'fit_eval' in kwargs:
            #    x_dat_comp = [xdata_eval for i in range(len(lbl_comp))]
            #else:
            #    x_dat_comp = [xdata for i in range(len(lbl_comp))]
            
            #end of retrieving possible info from kwargs
            
            # label raw data
            lbl_I = ['raw I', 'init-fit I', 'best-fit I']
            lbl_Q = ['raw Q', 'init-fit Q', 'best-fit Q']
            lbl_amp = ['raw Amp', 'init-fit Amp', 'best-fit Amp']
            lbl_phase = ['raw Phase', 'init-fit Phase', 'best-fit Amp']
            lbl_IQ = ['raw IQ', 'init-fit IQ', 'best_fit IQ']
            #lbl_comp0 = ['raw I', 'best-fit I']
            
            # comps items will be generated from fit
            
            #plot figures X and Y data
            fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                               2*cm_to_inch(wfig)))
            spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, hspace =0.1, wspace=0.1)
            ax0 = fig.add_subplot(spec[0, 0]) # I data vs fit
            ax1 = fig.add_subplot(spec[0, 1]) # Q data vs fit
            ax2 = fig.add_subplot(spec[1, 0]) # Amp data vs fit
            ax3 = fig.add_subplot(spec[1, 1]) # Phase data vs fit
            ax4 = fig.add_subplot(spec[0, 2]) # I vs Q fit
            #ax5 = fig.add_subplot(spec[1, 2]) # I breakdown + components
            
            # format for line_plot 
            # line_plot(ax, x_dat, y_dat, axs_lbl, lbl_arr, col_mark, ticks_2D, font_size)
            
            # create best-fit for first 4 plots
            # put title to only ax0
            ax0.set_title(axs_tit)
            line_plot(ax0, x_dat, i_dat, axs_lbl[:2], lbl_I, col_fit, 
                      ticks_null, font_size)
            ax0.legend(loc='best', frameon=False)
            
            line_plot(ax1, x_dat, q_dat, [axs_lbl[0], axs_lbl[2]], lbl_Q, 
                      col_fit, ticks_null, font_size)
            ax1.legend(loc='best', frameon=False)
            
            line_plot(ax2, x_dat, amp_dat, [axs_lbl[0], axs_lbl[3]], lbl_amp, 
                      col_fit, ticks_null, font_size)
            ax2.legend(loc='best', frameon=False)
            
            line_plot(ax3, x_dat, phase_dat, [axs_lbl[0], axs_lbl[4]], lbl_phase, 
                      col_fit, ticks_null, font_size)
            ax3.legend(loc='best', frameon=False)
            
            line_plot(ax4, i_dat, q_dat, [axs_lbl[1], axs_lbl[2]], lbl_IQ, 
                      col_fit, ticks_null, font_size)
            ax4.legend(loc='best', frameon=False)
            
            # breakdown in comps plot - otherwise rewrite line_plots for readability
            #line_plot(ax5, x_dat[:2], [i_dat[0], i_dat[2]], axs_lbl[:2], 
            #          lbl_comp0, [col_fit[0], col_fit[2]], 
            #          ticks_null, font_size)
            # line_plot(ax, x_dat, y_dat, axs_lbl, lbl_arr, col_mark, ticks_2D, font_size)
            #line_plot(ax5, x_dat_comp, yi_comp, axs_lbl[:2], 
            #          lbl_comp, [], ticks_null, font_size)
            #ax5.legend(loc='best', frameon=False, fontsize='small')
            plt.show()
            
        else: 
            
            # list parameters to be plotted
            y_dat = [ydata, init, out] #12, 12, 101 
            
            if 'fit_eval' in kwargs:
                #only out fit is modified
                x_dat = [xdata if i < 1 else xdata_eval for i in range(len(y_dat))]
            else:
                x_dat = [xdata for i in range(len(y_dat))]
            
            #name, comp = zip(*comps.items())
            #y_comp = list(comp)
            #lbl_comp = list(name)
            #if 'fit_eval' in kwargs:
            #    x_dat_comp = [xdata_eval for i in range(len(lbl_comp))]
            #else: 
            #    x_dat_comp = [xdata for i in range(len(lbl_comp))]
            
            # create parameters for fit
            lbl_fit = ['raw', 'init-fit', 'best-fit']
            #lbl_comp0 = ['raw', 'best-fit']
            
            # formulate data
            fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                               1*cm_to_inch(wfig)))
            spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
            
            ax0 = fig.add_subplot(spec[0, 0]) # data, initial and final fit
            #ax1 = fig.add_subplot(spec[1, 0]) # data and fit breakdown
            
            # plot 1
            ax0.set_title(axs_tit)
            line_plot(ax0, x_dat, y_dat, axs_lbl[:2], lbl_fit, col_fit, 
                      ticks_null, font_size)
            ax0.legend(loc='best', frameon=False)
            
            # breakdown of fits
            # different marker
            #line_plot(ax1, [x_dat[0],x_dat[2]], [y_dat[0],y_dat[2]], 
            #          axs_lbl[:2], lbl_comp0, col_fit, ticks_null, 
            #          font_size)
            # same marker
            #line_plot(ax1, x_dat_comp, y_comp, axs_lbl[:2], 
            #          lbl_comp, [], ticks_null, font_size)
            #ax1.legend(loc='best', frameon=False, ncols=4, fontsize='small')
            
            if 'axis_scale' in kwargs:
                ax0.set_xscale(kwargs['axis_scale'][0])
                ax0.set_yscale(kwargs['axis_scale'][1])
                
                #ax1.set_xscale(kwargs['axis_scale'][0])
                #ax1.set_yscale(kwargs['axis_scale'][1])
            plt.show()
    return

def show_anim_1Dplots():
    """
    Note: 
    contains animated data + fits on one-side, and animated extracted variables 
    [counted by axes on vars vs other axes]. Speeds up analysis prior to 
    replotting. Have save options for .gif or .mp4 (template is .gif for recheck)
    Useful for autofit_1D
    """
    return 

def show_iqcloud_report(dict_gmm, i_arr, q_arr, show=['N', 'N'], **kwargs):
    
    wfig=8.6
    
    #set axs_label
    
    if show[0] == 'Y':
        print('\n')
        #this report is important as it gives measure on further analysis the option of bundling
        print('results')
        # print(result.fit_report(min_correl=0.10))
        print('\n')
    
    if show[1] == 'Y':
        print('results')