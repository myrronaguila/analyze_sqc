# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:29:37 2024

TLS fit from literature (Klimov S2)
Reference: Figure S2 - Fitting Lorentzian Data (Barends)

Barends, R., ... & Martinis, J. M. (2013). Coherent Josephson qubit suitable 
for scalable quantum integrated circuits. Phys Rev Lett, 111(8), 080502. 
https://doi.org/10.1103/PhysRevLett.111.080502 

@author: Mai
"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from asqum_format import get_1D_plot_from_csv
from tls_spectrum import median_threshold_algo

#fitting parameters
from tls_spectrum import lm_nlz_const_fit
from tls_spectrum import lm_nlz_tls_fit
from tls_spectrum import loadmat_to_dict_1d

"""-------------------------Main--------------------------------------------"""
"""---Get Data from References----"""
dir2 = 'Barends/'
f1d2 = dir2 + 'TLS spectrum_Fig3c_Barends'

x_f1d2_GHz, y_f1d2_MHz = get_1D_plot_from_csv(f1d2)
wfig = 8.6

"""-----------Use median filter to select peak ranges-----------------------"""
lag_f1d2 = []
merge_f1d2 = []
lag_f1d2=[[5.25, 5.325], [5.39, 5.41], [5.55, 5.557], [5.566,5.568], [5.63,5.65], 
          [5.58, 5.61], [5.725, 5.75]] #selectively omit peaks
merge_f1d2=[[5.325,5.345], [5.347,5.368], [5.37,5.39], [5.42,5.435], [5.444,5.46],
            [5.466,5.481], [5.483, 5.5], [5.503,5.519], [5.522, 5.535], [5.537, 5.55],
            [5.555, 5.564], [5.57, 5.582], [5.608, 5.625], [5.65, 5.67], 
            [5.689, 5.7], [5.702, 5.715]] #16 peaks
#lag_f1d2 = [[5.45,5.47], [5.51, 5.52], [5.63, 5.64], [5.70,5.73]] #selective filter to remove candidate peaks
#merge_f1d2 = [[5.46,5.49], [5.57,5.59], [5.592, 5.605], [5.607, 5.63],  
#              [5.645, 5.662], [5.665,5.685], [5.69,5.73], [5.75, 5.8]] # select peak ranges to include
# get good range of filtered fit (one can tune best fit by increasing or decreasing number of points)
res_f1d2_med, dict_guess_f1d2 = median_threshold_algo(x_f1d2_GHz, y_f1d2_MHz, xlag=lag_f1d2, 
                                                      xmerge = merge_f1d2, threshold=1, 
                                                      num_pts=5, show='Y') #Barends

# # non-bundled data of lmfit, one must toggle the par_vary list for better fit
res_f1d2_best_fit, res_f1d2_params = lm_nlz_const_fit(x_f1d2_GHz, y_f1d2_MHz, dict_guess_f1d2, 
                                                      [False, True, True, True], show=['Y','Y'],
                                                      bundle = 'N', save=['Y',dir2 + 'Barends_best_fit'], 
                                                      time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])
# Elapsed time = 43.23 seconds

# save bundled data of lmfit
res_f1d2b_best_fit, res_f1d2b_params = lm_nlz_const_fit(x_f1d2_GHz, y_f1d2_MHz, dict_guess_f1d2, 
                                                        [False, True, True, True], show=['Y','Y'],
                                                        bundle = 'Y', save=['Y',dir2 + 'Barends_best_fit_B'], 
                                                        time='Y', input_info=['time_min', 0, ['pow_W',1.35]])
# Elapsed time = 42.98 seconds

# check lm_tls_function
# save non-bundled data with tls model - converts amplitude to coupling strength
res_f1d2_best_fit_2, res_f1d2_tls = lm_nlz_tls_fit(x_f1d2_GHz, y_f1d2_MHz, dict_guess_f1d2, 
                                                    [False, True, True, True], show=['Y','Y'],
                                                    bundle='N', save=['Y',dir2 + 'Barends_best_fit_2'], 
                                                    time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])
# Elapsed time = 46.58 seconds

# save bundled data with tls model - converts amplitude to coupling strength
res_f1d2b_best_fit_2, res_f1d2b_tls = lm_nlz_tls_fit(x_f1d2_GHz, y_f1d2_MHz, dict_guess_f1d2, 
                                                    [False, True, True, True], show=['Y','Y'],
                                                    bundle='Y', save=['Y',dir2 + 'Barends_best_fit_2B'], 
                                                    time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])
# Elapsed time = 44.4 seconds

"""---Load data from saved dictionary from .mat file for further processing---"""
d_bar_nlz_best_fit = loadmat_to_dict_1d(dir2 + 'Barends_best_fit')
d_bar_nlzb_best_fit = loadmat_to_dict_1d(dir2 + 'Barends_best_fit_B')
d_bar_tls_best_fit = loadmat_to_dict_1d(dir2 + 'Barends_best_fit_2')
d_bar_tlsb_best_fit = loadmat_to_dict_1d(dir2 + 'Barends_best_fit_2B')