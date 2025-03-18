# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:43:04 2025

Reproduce and test fit model for superconducting qubits
-> priority because of prof. Ke's paper. 
-> priority because of prof. Ke's paper.

CAS -> parameters based on supplementary information of Chinese Academy of Sciences

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters

from ls_qbt_cplr_volts_hz import f01_volts_to_hz
from ls_qbt_cplr_volts_hz import anticrossing_model
from ls_qbt_cplr_volts_hz import anticrossing_res2 #1D array 
from ls_qbt_cplr_volts_hz import anticrossing_res3 #check if this works

from lm_utils import get_1D_plot_from_csv
from lm_qbt_cplr_volts_hz_v1 import lm_f01_volts_to_hz

from lm_qbt_cplr_volts_hz_v1 import guess_f01_volts_to_hz
from lm_qbt_cplr_volts_hz_v1 import struct_anticrossing_data
from lm_qbt_cplr_volts_hz_v1 import lm_min2_anticrossing


from numpy import random

"""--------------------Parameters-Heng-Fan Parameters-----------------------"""
dir_hf = 'heng_fan_nat_com/'
file_q1 = dir_hf + 'nat_com_q1_zbias_vs_q1_freq'
# Dressed states
file_q2c2q3_c2d = dir_hf + 'nat_com_q2c2q3_c2_zbias_vs_c2_freq_Hz'
file_q2c2q3_q2d = dir_hf + 'nat_com_q2c2q3_c2_zbias_vs_q2_freq_Hz'
file_q2c2q3_q3d = dir_hf + 'nat_com_q2c2q3_c2_zbias_vs_q3_freq_Hz'
# Anticrossing spectrum
file_q2c2_anti = dir_hf + 'nat_com_q2c2_c2_zbias_vs_q2_freq_Hz'
file_q3c2_anti = dir_hf + 'nat_com_q3c2_c2_zbias_vs_q3_freq_Hz'

fq1_vs_Vq1 = get_1D_plot_from_csv(file_q1) # bare

# build saving method for parameters
f_Ec_q1 = 195.8E6 #Hz

"""--------------------------------Q1-Spectrum------------------------------"""
dict_g_q1 = guess_f01_volts_to_hz(xdata=fq1_vs_Vq1[0], 
                                  ydata=fq1_vs_Vq1[1], 
                                  f_Ec=f_Ec_q1)


guess_pars = [dict_g_q1['A_conv'], dict_g_q1['v0'], dict_g_q1['f_max'], 
              dict_g_q1['fact'], f_Ec_q1]

fit_fq1_hf, dict_fq1_hf= lm_f01_volts_to_hz(xdata=fq1_vs_Vq1[0], 
                                            ydata=fq1_vs_Vq1[1], 
                                            show=['N','N'],
                                            guess=guess_pars,
                                            bool_params=[True, True, True, True, False],
                                            device='Q1_hf',
                                            method='least_sq')

# # plot data
b1 = 1E-6
plt.title('Q1 Z-bias vs voltage')
plt.plot(fq1_vs_Vq1[0], fq1_vs_Vq1[1]*b1, 'k.', label='Q1_HF_data')
plt.plot(fit_fq1_hf[0], fit_fq1_hf[1]*b1, 'm-', label='Q1_HF_fit')
plt.xlabel('Q1 Z-bias (au)')
plt.ylabel('Q1 Frequency (MHz)')
#plt.xlim(-0.3,0.3)
#plt.ylim(4850,5500)
plt.show()

"""Fit is in good agreement and holds with the assumption that junction asymmetry = 1"""

"""---------------------------------Q2-C2-Q3 Spectrum-----------------------"""
fc2d_vs_Vc2 = get_1D_plot_from_csv(file_q2c2q3_c2d) # bare + dressed
fq2d_vs_Vc2 = get_1D_plot_from_csv(file_q2c2q3_q2d) #anti-crossing dressed
fq3d_vs_Vc2 = get_1D_plot_from_csv(file_q2c2q3_q3d) # anti-crossing dressed

f_Ec_q2 = 194.5E6 #Hz
f_Ec_q3 = 195.4E6 #Hz

# coupling strength in q2-c2-q3 system
g2_add = 0E6
g3_add = 0E6
g_q2c2_Hz = 84.13E6 + g2_add  #unknown
g_q3c2_Hz = 96.68E6 + g3_add #unknown
g_q2q3_Hz = 10.06E6 #unknown
f_Ec_c = 200E6 #Hz

# assumed anharmonicity for all couplers are Ec=200E6 Hz
# wq2 and wq3 normally set via Hz to voltage and voltage to Hz conversion.
fq2_add = 3.8E6 # there is an offset not of fq but of the idle frequency due to rounding errors
fq3_add = 9.0E6 # there #with g3_add=11E6
fq2_idle = 4760E6 + fq2_add 
fq3_idle = 5330E6 + fq3_add

"""
Known: fact=1; fq2_idle, fq3_idle (from Q2 / Q3 spectrum), d_C2=0 (fact=1)
Parameters to find for Q2C2Q3
For C2: fmax, A_conv, v0, assume fact=1; d=0

"""
dict_g_c2 = guess_f01_volts_to_hz(xdata=fc2d_vs_Vc2[0], 
                                  ydata=fc2d_vs_Vc2[1], 
                                  f_Ec=f_Ec_c) 

# note that due to incomplete characterization of the coupler spectrum, fact=0.98 is wrong.
# for incomplete spectra or unknown junction assymmetry, assume fact=1
# guess_pars_c2 = [dict_g_c2['A_conv'], dict_g_c2['v0'], dict_g_c2['f_max'], 
#                  dict_g_c2['fact'], f_Ec_c]
guess_pars_c2 = [dict_g_c2['A_conv'], dict_g_c2['v0'], dict_g_c2['f_max'], 
                  1, f_Ec_c]

"""For C2 and coupling, we use nelder-mead"""
fit_fc2_hf, dict_fc2_hf= lm_f01_volts_to_hz(xdata=fc2d_vs_Vc2[0], 
                                            ydata=fc2d_vs_Vc2[1], 
                                            show=['N','N'],
                                            guess=guess_pars_c2,
                                            bool_params=[True, True, True, False, False],
                                            device='C2_hf',
                                            method='nelder') # works best with nelder, but not leastsq

"""
f_Ejmax = 20,000,000,000.0 Hz => 20 GHz
f_Ejmax_lit = non.
"""

b1=1E-6
plt.title('C2 Z-bias vs voltage')
plt.plot(fc2d_vs_Vc2[0], fc2d_vs_Vc2[1]*b1, 'k.', label='C2_HF_data')
plt.plot(fq2d_vs_Vc2[0], fq2d_vs_Vc2[1]*b1, 'b.', label='Q2_HF_data')
plt.plot(fq3d_vs_Vc2[0], fq3d_vs_Vc2[1]*b1, 'r.', label='Q3_HF_data')
plt.plot(fit_fc2_hf[0], fit_fc2_hf[1]*b1, 'm-', label='C2_HF_fit')
plt.hlines(y=fq2_idle*b1,xmin=fc2d_vs_Vc2[0][0], 
            xmax=fc2d_vs_Vc2[0][-1], color='b', ls='dotted')
plt.hlines(y=fq3_idle*b1,xmin=fc2d_vs_Vc2[0][0], 
            xmax=fc2d_vs_Vc2[0][-1], color='r', ls='dotted')
plt.xlabel('C2 Z-bias (au)')
plt.ylabel('C2 Frequency (MHz)')
#plt.xlim(-0.3,0.3)
#plt.ylim(4850,5500)
plt.show()

"""-----------------------------Anti-crossing Data--------------------------"""
fq2c2_vs_Vc2 = get_1D_plot_from_csv(file_q2c2_anti) # anti-crossing
fq3c2_vs_Vc2 = get_1D_plot_from_csv(file_q3c2_anti) # anti-crossing

"""----------------------Show Q2C2 Anti-crossing spectra--------------------"""
f_add = 0 #fit is okay
A_add = 0 #fit is okay

print('\n')
print('fq2_idle = {:.5e} Hz'.format(fq2_idle))
print('fq2_mean = {:.5e} Hz'.format(np.average(fq2c2_vs_Vc2[1])))
# get average of minimum and maximum from guess_idle
Vc2_up = fq2c2_vs_Vc2[0][fq2c2_vs_Vc2[1] > fq2_idle] # characteristic of numpy, 34 elem
fq2c2_up = fq2c2_vs_Vc2[1][fq2c2_vs_Vc2[1] > fq2_idle] # characteristic of numpy, 34 elem
Vc2_down = fq2c2_vs_Vc2[0][fq2c2_vs_Vc2[1] < fq2_idle] # characteristic of numpy, 29 elem
fq2c2_down = fq2c2_vs_Vc2[1][fq2c2_vs_Vc2[1] < fq2_idle] # characteristic of numpy, 29 elem
fq2_guess = (np.amin(fq2c2_up)+np.amax(fq2c2_down))/2
print('fq2_guess = {:.5e} Hz'.format(fq2_guess))

"""
fq2_idle = 4.76380e+09 Hz
fq2_mean = 4.76422e+09 Hz => fq2_mean-fq2_idle= 0.00042
fq2_guess = 4.76266e+09 Hz => fq2_guess - fq2_idle = -0.00114 => worse analysis
"""

# simulate C2 frequency => for voltage to frequency conversion
fc2_q2c2_sim = f01_volts_to_hz(vpk=fq2c2_vs_Vc2[0], 
                               A_conv=dict_fc2_hf['A_conv'][0]+A_add, 
                               v0=dict_fc2_hf['v0'][0], 
                               f_max=dict_fc2_hf['f_max'][0]+f_add)

# simulate anti-crossing
fq2c2_anti_sim = anticrossing_model(f_tune=fc2_q2c2_sim, f_fixed=fq2_idle, 
                                    g_hz=g_q2c2_Hz)
fq2c2_anti_hi = fq2c2_anti_sim[:len(fq2c2_vs_Vc2[0])] # upper branch 
fq2c2_anti_lw = fq2c2_anti_sim[len(fq2c2_vs_Vc2[0]):] # lower branch

# reshuffle data based on upper and lower branch.
params = Parameters()
params.add_many(('f_fixed', fq2_idle),
                ('g_hz', g_q2c2_Hz))

"""---------------Trial 2, Q2C2 Anti-crossing fit with visible branches-----"""
# anticrossing_res2 good in sequential data but bad in fit.
fq2c2_anti_sim2 = anticrossing_res2(params, f_tune=fc2_q2c2_sim, 
                                    data=fq2c2_vs_Vc2[1], weights=None,
                                    output='model')

fc2_q2c2_com, fq2c2_anti_com, q2c2_branch = struct_anticrossing_data(fc2_q2c2_sim, 
                                                                      fq2c2_vs_Vc2[1],
                                                                      fq2_idle)

# test model, good in sorting global upper and lower branches but bad in sequential data
fq2c2_anti_sim3 = anticrossing_res3(params, f_tune=fc2_q2c2_com, 
                                    data=fq2c2_anti_com, branch=q2c2_branch, 
                                    weights=None,
                                    output='model')

# check data for separation of upper and lower plots.
plt.plot(fc2_q2c2_com[:34], fq2c2_anti_com[:34], 'k.', label='upper, data')
plt.plot(fc2_q2c2_com[34:], fq2c2_anti_com[34:], 'b.', label='lower, data')
plt.plot(fc2_q2c2_com[:34], fq2c2_anti_sim3[:34], 'g', label='upper, model')
plt.plot(fc2_q2c2_com[34:], fq2c2_anti_sim3[34:], 'm', label='lower, model')
plt.show()

# flexi-fit - failed fit. 20250313 -> More accurate fit due to improvement in guess.
fit_fq2c2_anti, dict_fq2c2_anti = lm_min2_anticrossing(xdata=fc2_q2c2_sim, 
                                                        ydata=fq2c2_vs_Vc2[1], 
                                                        show=['Y','Y'], 
                                                        guess=[fq2_idle,g_q2c2_Hz],
                                                        bool_params = [True, True],
                                                        device='Q2C2_hf',
                                                        method='nelder')

plt.title('C2 Z-bias vs voltage, Q2C2, plotting data and sim prior to fit')
plt.plot(fq2c2_vs_Vc2[0], fq2c2_vs_Vc2[1]*b1, 'k.', label='Q2C2_d_data')
plt.plot(fq2c2_vs_Vc2[0], fq2c2_anti_hi*b1, 'r:', label='Q2C2_hi_sim1')
plt.plot(fq2c2_vs_Vc2[0], fq2c2_anti_lw*b1, 'b:', label='Q2C2_lw_sim2')
plt.plot(fq2c2_vs_Vc2[0], fq2c2_anti_sim2*b1, 'm-', label='Q2C2_sim_res')
plt.plot(fq2c2_vs_Vc2[0], fc2_q2c2_sim*b1, color='tab:orange', ls='dashed', label='C2 Frequency')
plt.plot(fq2c2_vs_Vc2[0], fit_fq2c2_anti[1]*b1, 'g--', label='Q2C2_residual')
plt.hlines(y=fq2_idle*b1, xmin=np.amin(fq2c2_vs_Vc2[0]), xmax=np.amax(fq2c2_vs_Vc2[0]), 
            colors='k', ls='dashed')
plt.xlim(-2.5,2.5)
plt.ylim(np.amin(fq2c2_vs_Vc2[1])*b1, np.amax(fq2c2_vs_Vc2[1])*b1) # not appreciable due to constraint
plt.xlabel('C2 Z-bias (au)')
plt.ylabel('Q2 Frequency (MHz)')
plt.legend(loc='best')
plt.show()

"""
20250313
From fit (nelder):
    bool_params = [False, True]
[[Variables]]
    f_fixed:  4.7638e+09 (fixed)
    g_hz:     84339565.0 +/- 858953.794 (1.02%) (init = 1.516e+07)

bool_params = [True, True]
[[Variables]]
    f_fixed:  4.7647e+09 +/- 154372.751 (0.00%) (init = 4.76336e+09)
    g_hz:     85097289.8 +/- 697572.489 (0.82%) (init = 1.516e+07)

From literature:
    fq2_idle = 4.76380e+09 Hz
    g_q2c2_Hz = 84.13E6 + g2_add  #unknown
    
Analysis in good agreement.
"""

"""------Trial 3, Simulated Q2C2 Anti-crossing fit with visible branches-----"""
Vc2_sim = np.linspace(-2.5, 2.5, 101)
fc2_q2c2_sim2 = f01_volts_to_hz(vpk=Vc2_sim, 
                               A_conv=dict_fc2_hf['A_conv'][0], 
                               v0=dict_fc2_hf['v0'][0], 
                               f_max=dict_fc2_hf['f_max'][0])

# simulate anti-crossing with noise
fq2c2_anti_sim2 = anticrossing_model(f_tune=fc2_q2c2_sim2, f_fixed=fq2_idle, 
                                    g_hz=g_q2c2_Hz) + random.normal(loc=0,scale=0.5E6,size=2*len(Vc2_sim))
fq2c2_anti_hi2 = fq2c2_anti_sim2[:len(Vc2_sim)] # upper branch 
fq2c2_anti_lw2 = fq2c2_anti_sim2[len(Vc2_sim):] # lower branch

# test format for fit
xdata_test = np.concatenate((fc2_q2c2_sim2, fc2_q2c2_sim2))
#print(len(xdata_test[:101]))
#print(len(fq2c2_anti_sim2[:101]))
fit_fq2c2_test, dict_fq2c2_test = lm_min2_anticrossing(xdata=xdata_test, 
                                                       ydata=fq2c2_anti_sim2, 
                                                       show=['N','N'], 
                                                       guess=[fq2_idle,g_q2c2_Hz],
                                                       bool_params = [False, True],
                                                       device='Q2C2_hf',
                                                       method='nelder')
"""Note: Fit too sensitive to the average. Better that the fit be put not on
mean but in-between peaks"""

plt.title('Simulated Two-Peak fit')
plt.plot(Vc2_sim, fq2c2_anti_hi2*b1, 'r.', label='Q2C2_hi_sim1')
plt.plot(Vc2_sim, fq2c2_anti_lw2*b1, 'b.', label='Q2C2_lw_sim2')
plt.plot(Vc2_sim, fc2_q2c2_sim2*b1, color='tab:orange', ls='dashed', label='C2 Frequency_sim')
plt.plot(Vc2_sim, fit_fq2c2_test[1][:101]*b1, 'g--', label='Q2C2_residual, up')
plt.plot(Vc2_sim, fit_fq2c2_test[1][101:]*b1, 'm--', label='Q2C2_residual, down') 
plt.hlines(y=fq2_idle*b1, xmin=np.amin(Vc2_sim), xmax=np.amax(Vc2_sim), 
            colors='k', ls='dashed')
plt.xlim(-2.5,2.5)
plt.ylim(4.5E9*b1, 5.0E9*b1) # not appreciable due to constraint
plt.xlabel('C2 Z-bias (au)')
plt.ylabel('Q2 Frequency (MHz)')
plt.legend(loc='best')
plt.show()

"""
Conclusion: The fit works as long as the spectra gets truncated above or below its 
own frequencies. Anticrossing must not show big deviations. Also the initial
guess fit from the model does not show good agreement with the ending g_hz.
We implemented weights to improve the fit internally.


20250313 - For ideal data with more spots. using lm_min2_anticrossing
from fit: (nelder)
bool_params = [True, True]
[[Variables]]
    f_fixed:  4.7638e+09 +/- 46902.7581 (0.00%) (init = 4.762857e+09)
    g_hz:     83906333.0 +/- 115721.246 (0.14%) (init = 1.00425e+08)

bool_params = [False, True]
[[Variables]]
    f_fixed:  4.7638e+09 (fixed)
    g_hz:     83911870.8 +/- 115256.634 (0.14%) (init = 1.00425e+08)
None

"""

"""-----------------------Show Q2C3 Anti-crossing spectra-------------------"""
# coupler frequency in Hz

# get average of minimum and maximum from guess_idle
Vc3_up = fq3c2_vs_Vc2[0][fq3c2_vs_Vc2[1] > fq3_idle] # characteristic of numpy, 34 elem
fq3c2_up = fq3c2_vs_Vc2[1][fq3c2_vs_Vc2[1] > fq3_idle] # characteristic of numpy, 34 elem
Vc3_down = fq3c2_vs_Vc2[0][fq3c2_vs_Vc2[1] < fq3_idle] # characteristic of numpy, 29 elem
fq3c2_down = fq3c2_vs_Vc2[1][fq3c2_vs_Vc2[1] < fq3_idle] # characteristic of numpy, 29 elem
fq3_guess = (np.amin(fq3c2_up)+np.amax(fq3c2_down))/2
#print('fq3_guess = {:.5e} Hz'.format(fq3_guess))

"""
fq3_idle = 5.33900e+09 Hz
fq3_mean = 5.33391e+09 Hz => fq3_mean - fq3_idle = 0.00509e+09 # I guess np.average is okay
fq3_guess = 5.33584e+09 Hz => fq3_guess - fq3_idle = -0.00316 varying success in differences.
"""

# simulate fc2 frequency # not as linear as thought.
fc2_q3c2_sim = f01_volts_to_hz(vpk=fq3c2_vs_Vc2[0], 
                               A_conv=dict_fc2_hf['A_conv'][0]+A_add, 
                               v0=dict_fc2_hf['v0'][0], 
                               f_max=dict_fc2_hf['f_max'][0]+f_add)

# simulation of anti-crossing between q3 and c2
fq3c2_anti_sim = anticrossing_model(f_tune=fc2_q3c2_sim, f_fixed=fq3_idle, 
                                    g_hz=g_q3c2_Hz)
fq3c2_anti_hi = fq3c2_anti_sim[:len(fq3c2_vs_Vc2[0])] # upper branch model
fq3c2_anti_lw = fq3c2_anti_sim[len(fq3c2_vs_Vc2[0]):] # lower-branch branch model

fit_fq3c2_anti, dict_fq3c2_anti = lm_min2_anticrossing(xdata=fc2_q3c2_sim, 
                                                       ydata=fq3c2_vs_Vc2[1], 
                                                       show=['Y','Y'], 
                                                       guess=[fq3_idle,g_q3c2_Hz],
                                                       bool_params = [False, True],
                                                       device='Q3C2_hf',
                                                       method='nelder')

# optional
plt.title('C2 Z-bias vs voltage, Q3C2, plotting data and sim prior to fit')
plt.plot(fq3c2_vs_Vc2[0], fq3c2_vs_Vc2[1]*b1, 'k.', label='Q3C2_d_data')
plt.plot(fq3c2_vs_Vc2[0], fq3c2_anti_hi*b1, 'r:', label='Q3C2_hi_sim')
plt.plot(fq3c2_vs_Vc2[0], fq3c2_anti_lw*b1, 'b:', label='Q3C2_lw_sim')
plt.plot(fq3c2_vs_Vc2[0], fc2_q3c2_sim*b1, color='tab:orange', ls='dashed', label='C2 Frequency') 
plt.plot(fq3c2_vs_Vc2[0], fit_fq3c2_anti[1]*b1, 'g--', label='Q3C2_residual')
plt.hlines(y=fq3_idle*b1, xmin=np.amin(fq3c2_vs_Vc2[0]), xmax=np.amax(fq3c2_vs_Vc2[0]), 
            colors='k', ls='dashed')
plt.xlim(-2.5,2.5)
plt.ylim(np.amin(fq3c2_vs_Vc2[1])*b1, np.amax(fq3c2_vs_Vc2[1])*b1)
plt.xlabel('C2 Z-bias (au)')
plt.ylabel('Q3 Frequency (MHz)')
plt.legend(loc='best')
plt.show()

"""
from fit(nelder): 20250313
bool_params = [True, True]
[[Variables]]
    f_fixed:  5.3394e+09 +/- 84391.1076 (0.00%) (init = 5.334784e+09)
    g_hz:     97097164.0 +/- 369065.050 (0.38%) (init = 1.65977e+07)

bool_params = [False, True]
[[Variables]]
    f_fixed:  5.339e+09 (fixed)
    g_hz:     96462587.0 +/- 400828.990 (0.42%) (init = 1.65977e+07)

from literature:
    fq3_idle = 5.33900e+09 Hz
    g_q3c2_Hz = 96.68E6 #unknown

Good agreement
"""

"""-Further improvement would be the interface and coupling with a dressed anticrossing"""
import pipreqs
# pipreqs.main(["C:\\Users\\Mai\\Desktop\\Personal Progress Report\\Mai Analysis Package\\20240116-Volt_to_Freq_with_pulse_correct\\Github upload\\main_qbt_cplr_heng_fan_v1.py"])
# AttributeError: module 'pipreqs' has no attribute 'main'

import subprocess

subprocess.run(["pipreqs", "--force", "."])