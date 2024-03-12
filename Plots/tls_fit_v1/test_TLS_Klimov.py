# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:29:37 2024

TLS fit from literature (Klimov S2)
Reference: Figure S2 - Fitting Lorentzian Data (Klimov)

Klimov, P. V., Kelly, J., Chen, Z., Neeley, M., Megrant, A., Burkett, B., 
Barends, R., Arya, K., Chiaro, B., Chen, Y., Dunsworth, A., Fowler, A., 
Foxen, B., Gidney, C., Giustina, M., Graff, R., Huang, T., Jeffrey, E., 
Lucero, E., . . . Martinis, J. M. (2018). Fluctuations of Energy-Relaxation 
Times in Superconducting Qubits. Phys Rev Lett, 121(9), 090502. 
https://doi.org/10.1103/PhysRevLett.121.090502 

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
dir1 = 'Klimov/'
f1d1 = dir1 + 'TLS spectrum_S2_Klimov'

x_f1d1_GHz, y_f1d1_MHz = get_1D_plot_from_csv(f1d1)
wfig = 8.6

"""------------------Create Guess Algorithms for TLS peaks------------------"""
y_f1d1_base = np.ones(len(x_f1d1_GHz))*(np.average(y_f1d1_MHz)-np.std(y_f1d1_MHz)/4)

"""-----------Use median filter to select peak ranges-----------------------"""
lag_f1d1 = [[5.45,5.47], [5.51, 5.52], [5.63, 5.64], [5.70,5.73]] #selective filter
merge_f1d1 = [[5.46,5.49], [5.57,5.59], [5.592, 5.605], [5.607, 5.63], 
              [5.645, 5.662], [5.665,5.685], [5.69,5.73], [5.75, 5.8]]
# get good range of filtered fit 
res_f1d1_med, dict_guess_f1d1 = median_threshold_algo(x_f1d1_GHz, y_f1d1_MHz, xlag=lag_f1d1, 
                                                      xmerge = merge_f1d1, threshold=1, 
                                                      num_pts=11, show='Y') #Klimov

# non-bundled data of lmfit
res_f1d1_best_fit, res_f1d1_params = lm_nlz_const_fit(x_f1d1_GHz, y_f1d1_MHz, dict_guess_f1d1, 
                                                      [False, True, True, True], show=['Y','Y'],
                                                      bundle = 'N', save=['Y','Klimov_best_fit'], 
                                                      time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])

# save bundled data of lmfit
res_f1d1b_best_fit, res_f1d1b_params = lm_nlz_const_fit(x_f1d1_GHz, y_f1d1_MHz, dict_guess_f1d1, 
                                                        [False, True, True, True], show=['Y','Y'],
                                                        bundle = 'Y', save=['Y','Klimov_best_fit_B'], 
                                                        time='Y', input_info=['time_min', 0, ['pow_W',1.35]])

# check lm_tls_function
# save non-bundled data with tls model
res_f1d1_best_fit_2, res_f1d1_tls = lm_nlz_tls_fit(x_f1d1_GHz, y_f1d1_MHz, dict_guess_f1d1, 
                                                    [False, True, True, True], show=['Y','Y'],
                                                    bundle='N', save=['Y','Klimov_best_fit_2'], 
                                                    time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])

# save bundled data with tls model
res_f1d1b_best_fit_2, res_f1d1b_tls = lm_nlz_tls_fit(x_f1d1_GHz, y_f1d1_MHz, dict_guess_f1d1, 
                                                    [False, True, True, True], show=['Y','Y'],
                                                    bundle='Y', save=['Y','Klimov_best_fit_2B'], 
                                                    time='Y', input_info=['time_min', 0.5, 'pow_W', 1.35E-9])

"""---Load data from saved dictionary from .mat file for further processing---"""
d_kli_nlz_best_fit = loadmat_to_dict_1d(dir1 + 'Klimov_best_fit')
d_kli_nlzb_best_fit = loadmat_to_dict_1d(dir1 + 'Klimov_best_fit_B')
d_kli_tls_best_fit = loadmat_to_dict_1d(dir1 + 'Klimov_best_fit_2')
d_kli_tlsb_best_fit = loadmat_to_dict_1d(dir1 + 'Klimov_best_fit_2B')