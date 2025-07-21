# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:34:10 2025

crosstalk_fig_5v2, CZ bridge and leakage
 - with corrected data for voltage and population
 - with fitting in g_factor
 
ver2 - 20250418 => modification on error on transmon calculations
 - fixed qbt_cplr f01_from_transmon

20250424 => rushed to work out the figures according to flux quanta. 
(In reality) => impressions: Flux quanta may not be reliable since they are 
explicitly done via work-arounds.
(better would be to use the term CZ detuning and coupler detuning.)

20250502 => refer to fsim for simulated Rabi Model

20250511 => Changed the reference source with the help of li-chieh. 

20250529 => using fact=4.715, not working well from 1.

20250721 => upload to github for sharing
@author: Mai
"""

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os # get location of current script.
import time as time
"""For opening an experimental dataset"""
from xarray import open_dataset # assu

"""For curve fit and plotting"""
from data_plots_qm import cm_to_inch, set_font_default
from data_plots_qm import color_map

"""Analyze and map data"""
from lm_qbt_cplr_volts_hz import guess_f01_volts_to_hz # initiated fit
from lm_qbt_cplr_volts_hz import lm_f01_volts_to_hz # for fitting
from qbt_cplr_volts_hz import volts_to_norm_flux # for secondary axis conversion
from qbt_cplr_volts_hz import f01_volts_to_hz, g_qq_hz # not work on fit.

from lm_tdm import lm_wave_c
from scipy.io import loadmat
from qbt_cplr_volts_hz import czbr_freq_to_geff

# 20250514 - univariate spline
# from scipy.interpolate import UnivariateSpline
# from scipy.stats import zscore

# 20250515
from lm_qbt_cplr_volts_hz import guess_g_qcq_hz1, lm_min_gqcq_hz1 # no residual coupling

#20250516
from qbt_cplr_volts_hz import flip_sign_mod
from lm_qbt_cplr_volts_hz import guess_g_qcq_hz2, lm_min_gqcq_hz2 # with residual coupling

"""------------------------Upload data file---------------------------------"""
dir_main = os.getcwd()
dir_fig5 = '\\crosstalk_paper\\fig5_CZ_bridge_leakage\\'
# for replication
dir_fig5 = '\\dataset\\'

# uncal -2, 0.17
fname_q4q4_c3z_pbr = dir_main + dir_fig5 + 'crosstalk_q4q3_cz_coupler_c3_z_bridge.nc' 
# obtain working data
q4q3c3z_pbr = open_dataset(fname_q4q4_c3z_pbr)
x_q4q3c3z_pbr = q4q3c3z_pbr['q0_ro']['amps'].values
idx_1 = np.where(x_q4q3c3z_pbr == -2)[0][0] #idx_1 = 0
idx_2 = np.where(np.round(x_q4q3c3z_pbr,3) == 0.170)[0] #idx_2 = 868

# gs0, CZ_bridge
fname_q4q3_c3z_ubr = dir_main + dir_fig5 + 't_crosstalk_q4q3_cz_coupler_c3_z_bridge.nc' #row 0, drive
fname_q4q3_c3z_cbr = dir_main + dir_fig5 + 't_compense_q4q3_cz_coupler_c3_z_bridge.nc' #row 1, drive

#q4q3_c3, source/crosstalk_z:c2, probe/detector_z: q3, q3_ro
q4q3c3z_ubr = open_dataset(fname_q4q3_c3z_ubr)
#sq3pc2, source/crosstalk_z:q3, probe/detector_z: c2/q7, q7_ro
q4q3c3z_cbr = open_dataset(fname_q4q3_c3z_cbr)

# gs1, CZ_leak
fname_q4q3_c3z_ulk = dir_main + dir_fig5 + 't_crosstalk_q4q3_cz_coupler_c3_z_leakage.nc' #row 0, drive
fname_q4q3_c3z_clk = dir_main + dir_fig5 + 't_compense_q4q3_cz_coupler_c3_z_leakage.nc' #row 1, drive

#q4q3_c3, source/crosstalk_z:c2, probe/detector_z: q3, q3_ro
q4q3c3z_ulk = open_dataset(fname_q4q3_c3z_ulk)
#sq3pc2, source/crosstalk_z:q3, probe/detector_z: c2/q7, q7_ro
q4q3c3z_clk = open_dataset(fname_q4q3_c3z_clk)

"""-----------------------------------Check .nc dataset---------------------"""
print(q4q3c3z_ubr.data_vars.items())
"""
ItemsView(Data variables:
    q0_ro    (mixer, amps, time) float64 ...
    q1_ro    (mixer, amps, time) float64 ...)
time (x), qubit
amps (y), qubit
"""
print(q4q3c3z_ulk.data_vars.items())
"""
    q0_ro    (mixer, c_amps, amps) float64 ...
    q1_ro    (mixer, c_amps, amps) float64 ...)
c_amps (coupler amp)
amps (qubit)
"""

# rotation
k1=1

# q0_ro, Q3, qubit flux is coupler Q3, ubr - uncompensated bridge
x_q4q3c3z_ubr = q4q3c3z_ubr['q0_ro']['amps'].values #1000
# 20250511 => calibrated signal => [-0.36643, 0.13307] # this is right.
y_q4q3c3z_ubr = q4q3c3z_ubr['q0_ro']['time'].values #96
# 20250511 => [16, 396]
I_q4q3c3z_ubr = q4q3c3z_ubr['q0_ro'][0].values #(1000,96)
I_q4q3c3z_ubrt = np.flipud(np.rot90(I_q4q3c3z_ubr,k=k1)) # right direction #(96,1000)
# I_q4q3c4z_cbrt = (96,1000)

# q0_ro, Q3, uncompensated, cbr - compensated bridge
x_q4q3c3z_cbr = q4q3c3z_cbr['q0_ro']['amps'].values #1000
# 20250511 => calibrated signal => [-0.36643, 0.13307]
y_q4q3c3z_cbr = q4q3c3z_cbr['q0_ro']['time'].values #96
# 20250511 => [16, 396]
I_q4q3c3z_cbr = q4q3c3z_cbr['q0_ro'][0].values #(1000,96)
I_q4q3c3z_cbrt = np.flipud(np.rot90(I_q4q3c3z_cbr,k=k1)) # right direction #(96,1000)
# I_q4q3c4z_cbrt = (96,1000)

# q0_ro, Q3, qubit flux is coupler Q4 leakage, (100,100)
x_q4q3c3z_ulk = q4q3c3z_ulk['q0_ro']['amps'].values #x-axis, 100
# calibrated signal => [-0.123864, -0.044664], same
y_q4q3c3z_ulk = q4q3c3z_ulk['q0_ro']['c_amps'].values # y-axis, 100
# calibrated signal => [0.05357, 0.07337], same
I_q4q3c3z_ulk = q4q3c3z_ulk['q0_ro'][0].values #(100,100)
I_q4q3c3z_ulkt = np.flipud(np.rot90(I_q4q3c3z_ulk,k=k1)) # right direction #(100,100)

# q0_ro, Q3, (100, 100), Q4 being tuned
x_q4q3c3z_clk = q4q3c3z_clk['q0_ro']['amps'].values #(100,100)
# calibrated signal => [-0.123864, -0.044664] same
y_q4q3c3z_clk = q4q3c3z_clk['q0_ro']['c_amps'].values #(100,100)
# calibrated signal => [0.05357, 0.07337]
I_q4q3c3z_clk = q4q3c3z_clk['q0_ro'][0].values #(100,100)
I_q4q3c3z_clkt = np.flipud(np.rot90(I_q4q3c3z_clk,k=k1)) # right direction #(100,100)

"""-----Import calibration dataset extracted from fig5s---------------------"""

"""------Compare A-conv of previous dataset - assumed to be identical-------"""
# Qubit from Fig 2
dict_fq3 = loadmat(dir_main + dir_fig5 + 'dict_q3_f01_vs_vpk.mat') # check better dataset
dict_fq4 = loadmat(dir_main + dir_fig5 + 'dict_q4_f01_vs_vpk.mat') # check better dataset

# Coupler from Fig2
dict_fc2 = loadmat(dir_main + dir_fig5 + 'dict_c2_f01_vs_vpk.mat') # check better dataset
dict_fc3 = loadmat(dir_main + dir_fig5 + 'dict_c3_f01_vs_vpk.mat') # check better dataset

"""---20250328 -Update - the data here is not representative of Q3-C3-Q4
from fig2-4. Will be waiting for Nien-Yu for data. In the
meanwhile, we prepare the figure for scanning. But hopefully, Nien-Yu has
enough datasets to calibrate this method------------------------------------"""


dir_fig5s = '\\crosstalk_paper\\fig5_CZ_bridge_leakage\\valid_calibration\\'
# 20250721 - replication
dir_fig5s = '\\dataset\\'


"""--------------------------20250530---------------------------------------"""
# Qubits
dict_fq3s = loadmat(dir_main + dir_fig5s + 'dict_q3s_f01_vs_vpk_0530.mat') # asymmetric junction
dict_fq4s = loadmat(dir_main + dir_fig5s + 'dict_q4s_f01_vs_vpk_0530.mat') # asymmetric junction
# Couplers
dict_fc2s = loadmat(dir_main + dir_fig5s + 'dict_c2s_f01_vs_vpk_0530.mat') # symmetric junction
dict_fc3s = loadmat(dir_main + dir_fig5s + 'dict_c3s_f01_vs_vpk_0530.mat') # symmetric junction

"""Note - calibration for frequency here is totally different from fig2-4"""
f_Ec_q3s = dict_fq3s['f_Ec'][0][0] #204.7 MHz, make this constant as this is different
f_Ec_q4s = dict_fq4s['f_Ec'][0][0] #201.996 MHz, more likely used for CZ 02
f_Ec_c3s = dict_fc3s['f_Ec'][0][0] #225.6 MHz
# f_Ec_c3s = 127.2E6 #MHz - based on draft, 20250411
f_Ec_c2s = dict_fc3s['f_Ec'][0][0] #225.6 MHz
# f_Ec_c2s = 119.8E6 #MHz - based on draft, 20250411
# q0 = q3, q1 = q4, cz0 = x3
"""offset voltages for cz bridge / leakage. Otherwise there is a Y axis, offsets 
are there."""

vq3_ofs = -0.17504935319 # listed offset not corresponding to v0 via fit
vq4_ofs = -0.08386404608 # listed offset not via fit.
vc3_ofs = -0.14413 # listed offset not via fit

# offset listed but different configuration during measurement, suspecting vc2 adds stray complexity in CZ
vc2_ofs = -0.21335 # offset voltage using the whole process


"""
From additional database (dict_fq3s)
f01_q3_max = 4.7197 +/- 3E-5 GHz; A_conv = 0.666 +/- 0.002 /V; 
v0_q3 = -0.0287 +/- 0.0007; fact=1 => d=0; Ec_q3 = 204.7 MHz
fq3_ofs_br = 4708043936.985694 = 4708.0439 MHz

Compare with fig2 values of database  (dict_fq3)
f01_q3_max = 4.7669 +/- 2E-5 GHz; A_conv = 0.919 +/- 0.001 /V; 
v0_q3 = 0.0192 +/- 0.0003; fact=1 => d=0; Ec_q3 = 204.7 MHz

"""

fq3_ofs_br = f01_volts_to_hz(vpk=vq3_ofs, A_conv=dict_fq3s['A_conv'][0][0],
                             v0=dict_fq3s['v0'][0][0], 
                             f_max = dict_fq3s['f_max'][0][0], 
                             fact=dict_fq3s['fact'][0][0], f_Ec=f_Ec_q3s)

"""
From additional database (dict_fq4s)
f01_q4_max = 4.9308 +/- 3 GHz; A_conv = 0.3196 +/- 0.0007 /V; 
v0_q4 = -0.041 +/- 0.001; fact=1 => d=0; Ec=201.996E6 # original
fq4_ofs_br = 4930605846.094987 = 4930.6058 MHz

Compare with fig2 values of database (dict_fq4)
f01_q4_max = 5.080 +/- 2E-5 GHz; A_conv = 0.954 +/- 0.001 /V; 
v0_q4 = -0.0071 +/- 0.0003; fact=1 => d=0; Ec_q4 = 205.7 MHz
"""

fq4_ofs_br = f01_volts_to_hz(vpk=vq4_ofs, A_conv=dict_fq4s['A_conv'][0][0],
                             v0=dict_fq4s['v0'][0][0], 
                             f_max = dict_fq4s['f_max'][0][0], 
                             fact=dict_fq4s['fact'][0][0], f_Ec=f_Ec_q4s)

"""
From additional database (dict_fc3s)
f01_c3_max = 7.4927 +/- 3E-5 GHz; A_conv = 0.46629 +/- 0.001 / V; 
v0_c3 = -0.0654 +/- 0.0008; fact=1 => d=0; Ec=225.6E6 # original

# 150 MHz probe in frequency

Compare with fig2 values of database (dict_fc3)
f01_c3_max = 7.9091 +/- 3E-5 GHz; A_conv = 1.23619 +/- 0.00235 V; 
v0_c3 = 0.04733 +/- 0.0002; fact=1 => d=0; Ec_q3 = 225.6 MHz

"""
fc3_tune_br = f01_volts_to_hz(vpk=x_q4q3c3z_cbr, A_conv=dict_fc3s['A_conv'][0][0],
                         v0=dict_fc3s['v0'][0][0],
                         f_max = dict_fc3s['f_max'][0][0], 
                         fact=dict_fc3s['fact'][0][0], f_Ec=f_Ec_c3s)

"""Seems that fc3 suffers from bad quality datasets, otherwise, there is no
coupling - we modify the slope for A_conv and offset and compare this with
the extracted coupling strength. -> not real coupling but a modified coupling

Offset used based on detuned frequencies from CZ bridge, obtained from detuning
"""

# check CZ-detune, compare from previous qubit c3
fc3_tune_brm = f01_volts_to_hz(vpk=x_q4q3c3z_cbr, A_conv=dict_fc3['A_conv'][0][0] + 0.2,
                          v0=-0.14839162968280548,
                          f_max = dict_fc3s['f_max'][0][0], 
                          fact=dict_fc3s['fact'][0][0], f_Ec=f_Ec_c3s)

"""-Check database of c2s
From additional database (dict_fc2s)
f01_c2_max = 7.74239 +/- 4E-5 GHz; A_conv = 0.4600 +/- 0.002 / V; 
v0_c2 = -0.0830685 +/- 0.001; fact=1 => d=0; Ec=225.8E6 # original, Same as C2 of fig2.

# 150 MHz probe in frequency

Compare with fig2 values of database (dict_fc2)
f01_c2_max = 8.04377 +/- 1.8E-5 GHz; A_conv = 1.2663 +/- 0.0024 / V; 
v0_c2 = 0.0446 +/- 0.0002; fact=1 => d=0; Ec_q3 = 225.8 MHz

20250410 - Assessment on stray peak at fc3, different peak based on coupling.

At offset=-0.21335 V, the frequency is 7.735238105.209536 GHz,
way above c3. =>
"""

fc2_ofs_br = f01_volts_to_hz(vpk=vc2_ofs, A_conv=dict_fc2s['A_conv'][0][0],
                             v0=dict_fc2s['v0'][0][0], 
                             f_max = dict_fc2s['f_max'][0][0], 
                             fact=dict_fc2s['fact'][0][0], f_Ec=f_Ec_c2s)

"""Useful for checking frequency targetting"""

norm_flux_c3_br = volts_to_norm_flux(vpk=x_q4q3c3z_cbr, 
                                     A_conv=dict_fc3s['A_conv'][0][0], 
                                     v0=dict_fc3s['v0'][0][0])
# 20250511 => [-0.14034, 0.0925725]
# [-1.01299, 1.33222] - flux quanta

# norm_flux_c3_brm = volts_to_norm_flux(vpk=x_q4q3c3z_cbr, 
#                                      A_conv=dict_fc3['A_conv'][0][0], 
#                                      v0=-0.8675)

# norm_flux_c3_brm = volts_to_norm_flux(vpk=x_q4q3c3z_cbr, 
#                                      A_conv=dict_fc3['A_conv'][0][0], 
#                                      v0=-0.90)

"""-----Verify parameters for CZ from original datasets---------------------"""
# dir_fig5a = '\\crosstalk_paper\\fig5_CZ_bridge_leakage\\valid_calibration\\'


"""------------Plot interaction of frequencies between Q3, Q4, and C3-------"""
idle_arr = np.ones(len(x_q4q3c3z_cbr))

fc2_ofs_bra = idle_arr*fc2_ofs_br #c2_01 transition, in principle this should not appear but we check
fq3_ofs_bra = idle_arr*fq3_ofs_br #q3_01 transition
fq4_ofs_bra = idle_arr*fq4_ofs_br #q4_01 transition
fq4_12 = fq4_ofs_br-f_Ec_q4s

q4q3_sw_detune = fq4_ofs_br - fq3_ofs_br
q4q3_cz_detune = fq4_12 - fq3_ofs_br #q4 => |12> state, q3 => |01> state 
# previous setting using q4q3 br 20250411
print(q4q3_cz_detune) #20,565,909.109292984, detuning of fq4_12 and fq3_01
print(q4q3_sw_detune) #222561909.10929298 MHz
# after setting using q4q3 br 20250511
# 20565909.109292984
# 222561909.10929298
"""# does not converge using fsim if we use fc3 stats"""

# after calibration by z-offset, q4q3_cz_detune = 

b1 = 1E-6 # convert Hz to MHz
plt.title('Coupler Spectrum for CZ-SWAP')
plt.plot(x_q4q3c3z_cbr, fq3_ofs_bra*b1, 'r-', label='Q3s, f01_idle')
plt.plot(x_q4q3c3z_cbr, fq4_ofs_bra*b1, 'b-', label='Q4s, f01_idle')
plt.plot(x_q4q3c3z_cbr, (fq4_ofs_bra-f_Ec_q4s)*b1, 'b:', label='Q4s, f12')
plt.plot(x_q4q3c3z_cbr, fc3_tune_br*b1, 'k-', label='C3s for CZ Swap')  
plt.plot(x_q4q3c3z_cbr, fc3_tune_brm*b1, 'm:', label='C3_mod for CZ Swap')
plt.plot(x_q4q3c3z_cbr, fc2_ofs_bra*b1, 'g-', label='C2s, f01_idle')
plt.xlabel(r'C3 Flux Bias (V)')
plt.ylabel(r'$f_{01}$ (MHz)')
plt.legend(loc='best')
plt.show()

"""---------Analyze magnitude of coupling strength using lmwave-------------"""
start_time = time.time()
freq_fit_ubr = np.zeros(len(x_q4q3c3z_cbr))
freq_fit_cbr = np.zeros(len(x_q4q3c3z_cbr))
phase_fit_ubr = np.zeros(len(x_q4q3c3z_cbr)) # try to unwrap this
phase_fit_cbr = np.zeros(len(x_q4q3c3z_cbr)) # try to unwrap this
# sin fit best fitted the trend
# for i in range(len(x_q4q3c3z_ubr)):
#     dict_ubr = lm_wave_c(xdata=y_q4q3c3z_ubr*1E-9, 
#                           ydata=I_q4q3c3z_ubrt[:,i], 
#                           show=['N','N'], wave='sin')[1]
#     dict_cbr = lm_wave_c(xdata=y_q4q3c3z_cbr*1E-9, 
#                           ydata=I_q4q3c3z_cbrt[:,i], 
#                           show=['N','N'], wave='sin')[1]
#     freq_fit_ubr[i]=dict_ubr['sin_frequency'][0]
#     phase_fit_ubr[i]=dict_ubr['sin_shift'][0]
#     freq_fit_cbr[i]=dict_cbr['sin_frequency'][0]
#     phase_fit_cbr[i]=dict_cbr['sin_shift'][0]
"""Extracting datasets take 234 s."""

# # Estimating effective coupling as half of the fourier frequency.
freq_fit_ubr = np.load(dir_main + dir_fig5 + 'freq_fit_ubr.npy')
#freq_fit_ubr /= 2 # half of the fourier frequency. # assuming detuning = 0
freq_fit_cbr = np.load(dir_main + dir_fig5 + 'freq_fit_cbr.npy')
#freq_fit_cbr /= 2 # half of the fourier frequency. # assuming detuning = 0

# phase_fit_ubr = np.load(dir_main + dir_fig5 + 'phase_fit_ubr.npy')
# phase_fit_cbr = np.load(dir_main + dir_fig5 + 'phase_fit_cbr.npy')
# phase_fit_ubrw = np.unwrap(phase_fit_ubr)
# phase_fit_cbrw = np.unwrap(phase_fit_cbr)

"""Convert freq_fit_cbr after considering detuning effects"""
freq_fit_ubrm = czbr_freq_to_geff(freq_hz=freq_fit_ubr, det_hz=q4q3_cz_detune)
# # effective geff with zz interaction
freq_fit_cbrm = czbr_freq_to_geff(freq_hz=freq_fit_cbr, det_hz=q4q3_cz_detune)

# get sign of frequency for complete model


# plt.plot(x_q4q3c3z_ubr, freq_fit_ubr, 'k.', label='ubr')
# plt.plot(x_q4q3c3z_cbr, freq_fit_cbr, 'g.', label='cbr')
# plt.xlabel('Volts')
# plt.ylabel('Frequency (Hz)')
# plt.show()

# plt.plot(x_q4q3c3z_ubr, phase_fit_ubr, 'k.', label='ubr')
# plt.plot(x_q4q3c3z_cbr, phase_fit_cbr, 'g.', label='cbr')
# plt.xlabel('Volts')
# plt.ylabel('Phase Shift (rad)')
# plt.legend(loc='best')
# plt.show()

# dt_br = np.mean(np.diff(x_q4q3c3z_cbr))
# inst_freq_ubr = np.mean(np.gradient(phase_fit_ubr,dt_br)) / (2 * np.pi)
# inst_freq_cbr = np.mean(np.gradient(phase_fit_cbr,dt_br)) / (2 * np.pi)

# plt.plot(x_q4q3c3z_ubr, inst_freq_ubr, 'k.', label='ubr')
# plt.plot(x_q4q3c3z_cbr, inst_freq_cbr, 'g.', label='cbr')
# plt.xlabel('Volts')
# plt.ylabel('Phase / Volts')
# plt.legend(loc='best')
# plt.show()

"""
Presence of accumulated phase due to time distortions at long time scales
could not be suppressed. We think this is due to the non-superconducting
capability of the material, and inability to fully suppress them at long time
scales.
"""

"""------Fit the flux-frequency dataset to determine C3 offset--------------"""

# # Search for maximum frequency between C3 and get the applied voltage
# freq_fitc_max = freq_fit_cbrm[120:781].max() #43,054,106 Hz
# # print(freq_fitc_max) #43,054,106 Hz
# idx_freq_fit_c3 = np.where(freq_fit_cbrm == freq_fitc_max)[0][0] #453

# vpk_c3_ofs_czbr = x_q4q3c3z_cbr[idx_freq_fit_c3]
# print(vpk_c3_ofs_czbr)  #-0.8675 V

"""-------Extract offset of coupler due to computational state--------------"""

idxa, idxb = 80, 787 # need to extract new offset voltage.
# idxa, idxb = 80, 750 # need to extract new offset voltage.
# idxa, idxb = 0, 1001
# idxa, idxb = 73, 788 # first peak
# idxa, idxb = 73, 793 # first peak, large, right hand, 792
# idxa, idxb = 71, 795# second peak, large, right hand, 793
# idxa, idxb = 69, 797 # with plateau 

# grab offset from a semi-parabolic plot

dict_g = guess_f01_volts_to_hz(xdata=x_q4q3c3z_cbr[idxa:idxb], 
                                  ydata=freq_fit_cbrm[idxa:idxb], 
                                  f_Ec=2E5)

guess_g = [dict_g['A_conv'], dict_g['v0'], dict_g['f_max'], 
            1, 2E5]
method='nelder'
method='leastsq'
test_x = x_q4q3c3z_cbr[idxa:idxb]
test_y = freq_fit_cbrm[idxa:idxb]
# 20250523 - not sensitive enough / flexible enough to deal with new datasets
# works for fact_q3/q4 = 1
fit_g, dict_g= lm_f01_volts_to_hz(xdata=test_x, 
                                      ydata=test_y, 
                                      show=['Y','Y'],
                                      bool_params=[True, True, True, False, False],
                                      guess=guess_g,
                                      device='C3',
                                      method=method,
                                      save=['N', ''])

# 20250529 => fact_20250523 -> not working
fit_g, dict_g= lm_f01_volts_to_hz(xdata=test_x, 
                                      ydata=test_y, 
                                      show=['F','F'],
                                      bool_params=[False, True, True, False, False],
                                      guess=guess_g,
                                      device='C3',
                                      method=method,
                                      save=['N', ''])

# 20250530 => right database
fit_g, dict_g= lm_f01_volts_to_hz(xdata=test_x, 
                                      ydata=test_y, 
                                      show=['Y','Y'],
                                      bool_params=[True, True, True, False, False],
                                      guess=guess_g,
                                      device='C3',
                                      method=method,
                                      save=['N', ''])

# let us stop 20250529. Within the measurement ranges, U shape rather than parabolic shape.

# idxa, idxb = 0, 1001

# # offset is around -0.9060950 +/- 0.0009761822482662633 
# # update fc3 to adopt with Czbridge
# A_conv_c2 = 1.26633 #C2, weirdly enough, the slope of c2 works
# A_conv_c3 = dict_fc3['A_conv'][0][0] #C3, 1.2362
# A_conv_c3 = 0.97 #C3, 1.2362
# A_conv_c3 = 0.98
# A_conv_c3 = dict_fc3s['A_conv'][0][0] # 4.69512 +/- 0.0128697
# 20250511
A_conv_c3 = dict_fc3s['A_conv'][0][0] + 0.2 # looks good
# 20250530 - same
A_conv_c3 = dict_fc3s['A_conv'][0][0] + 0.2# looks good
# A_conv_c3 = dict_fc3s['A_conv'][0][0] # looks good
# A_conv_c3 = dict_fc3s['A_conv'][0][0]
# A_conv_c3 = (1.26633+1.2362)/2 # mean just to represent c2 and c3, more representative
fc3_tune_brm = f01_volts_to_hz(vpk=x_q4q3c3z_cbr, A_conv=A_conv_c3,
                               v0=dict_g['v0'][0],
                               f_max = dict_fc3s['f_max'][0][0], 
                               fact=dict_fc3s['fact'][0][0], f_Ec=f_Ec_c3s)

# dict_g['v0'] after correction => -0.14839162968280548

# Update norm flux c3_brm
norm_flux_c3_brm = volts_to_norm_flux(vpk=x_q4q3c3z_cbr, 
                                      A_conv=A_conv_c3, 
                                      v0=dict_g['v0'][0])
# [-1.06766, 1.37745], A_conv_c3 = 4.895123

# # for leakage
norm_flux_q4_lk = volts_to_norm_flux(vpk=x_q4q3c3z_clk, 
                                      A_conv=dict_fq4s['A_conv'][0][0],
                                      v0=0) # sounds like the measured data has no offset.

# 20250521, with offset considerations, not symmetric
v0_fq4s = dict_fq4s['v0'][0][0] # -0.0879969 +/- 0.000121716
# v0_fq4s = dict_fq4s['v0'][0][0] + 0.000121716 # -0.0879969
# v0_fq4s = dict_fq4s['v0'][0][0] + 0.0033 # -0.0879969

norm_flux_q4_lk = volts_to_norm_flux(vpk=x_q4q3c3z_clk, 
                                      A_conv=dict_fq4s['A_conv'][0][0],
                                      v0=v0_fq4s)

# # seems that leakage experiments have issues.

norm_flux_c3_lk = volts_to_norm_flux(vpk=y_q4q3c3z_clk, 
                                      A_conv=A_conv_c3, 
                                      v0=dict_g['v0'][0])

"""Assessment the measured data between CZ SWAP and Leakage and 
initial frequency spectroscopy has a big issue in offet and calibration."""

"""-----------------Simulate coupling strength for coupler-------------------"""
zz_g = 6E5 #zzg not much difference when zz_g = 0. follows trend well albeit with reduced amplitude
g_qqg = 36E6
g_ccg = 200E6 # assume symmetric g1c and g2c
detune = q4q3_cz_detune #CZ, 20 MHz
detune = q4q3_sw_detune #iSWAP, 222,561,909 = 222.6 MHz
#20250511 => detuning is cz detune rather than q4q3 detune
detune = q4q3_cz_detune

"""--Check new plot for fc3 using a shifted plot-and compare coupling with dispersion"""
b1 = 1E-6 # convert Hz to MHz

fig = plt.figure(constrained_layout=True)

gs = gridspec.GridSpec(1, 1, figure=fig)  # Adjusting spacing, two-column page strip
ax0 = fig.add_subplot(gs[0, 0]) # bridge prior to correction, with title
ax0t = ax0.twinx() # bridge after correction, with title

ax0.set_title('Coupler Spectrum for CZ-SWAP')
ax0.plot(x_q4q3c3z_cbr, fq3_ofs_bra*b1, 'r-', label='Q3s, f01')
ax0.plot(x_q4q3c3z_cbr, fq4_ofs_bra*b1, 'b-', label='Q4s, f01')
ax0.plot(x_q4q3c3z_cbr, (fq4_ofs_bra-f_Ec_q4s)*b1, 'b:', label='Q4s, f12')
ax0.plot(x_q4q3c3z_cbr, fc3_tune_br*b1, 'k-', label='C3s for CZ Swap')  
# ax0.plot(x_q4q3c3z_cbr, fc3_tune_brm*b1, 'c:', label='C3_mod for CZ Swap')
# 20250411 => [-2, 0.5]
# ax0.hlines(y=5985, xmin=-2, xmax=0.5, color='g', ls='dashed') # peak comes from resonator
# 20250511 => [-0.36643, 0.13307]
ax0.hlines(y=5985, xmin=-0.36643, xmax=0.13307, color='g', ls='dotted') # peak comes from resonator
ax0.set_ylim(4500,5000) # before 20250420
ax0.set_ylim(4500,7500) # after 20250420
ax0.set_xlabel(r'C3 Flux Bias (V)')
ax0.set_ylabel(r'$f_{01}$ (MHz)')
# ax0.legend(loc='best')

"""
20250418
Check absolute data of model according to capacitance calculations and 
estimations - based on si_capmat.py
"""

"""Initial guess based on lumped model"""
g_q3q4d = 6.0678E6 
g_q3c3d = 68.737E6
g_q4c3d = 71.512E6

"""Updated guess based on scaling"""
# 20250411 => working for [-2, 0.5]
c1 = 1
g_qqg = g_q3q4d *5.52 *c1 # 33.494 MHz
g_ccg = np.sqrt(g_q3c3d*g_q4c3d) *2.7*c1 # 189.299 MHz = 190 MHz.


# 20250511 => previous setting under bias not working, [-2, 0.5]
# changed dataset due to correction in fc3_br data based on what he gave.
c1 = 1
g_qqg = g_q3q4d * 6 * c1 # 33.494 MHz
g_ccg = np.sqrt(g_q3c3d*g_q4c3d) * 2.7 *c1 # 189.299 MHz = 190 MHz.

g_qqg = g_q3q4d * 5.8 * c1 # 33.494 MHz
g_ccg = np.sqrt(g_q3c3d*g_q4c3d) * 2.6 *c1 # 189.299 MHz = 190 MHz.

c1 = 1 # for detune = cz_detune
g_qqg = g_q3q4d * 6.1 * c1 # 37,013,580.0 MHz = 37 MHz
g_ccg = np.sqrt(g_q3c3d*g_q4c3d) * 2.9 *c1 # 196310161.4714837 MHz = 196.3 MHz

# 20250514, good for idx=0, 1001
c1 = 0.8
g_qqg = g_q3q4d * 5.2* c1 # 25.242 MHz = 37 MHz
g_ccg = np.sqrt(g_q3c3d*g_q4c3d) * 2.85*c1 # 159.85 MHz = 196.3 MHz

# for narrow coupling, good for idx=80, 787
# c1 = 0.75
# g_qqg = g_q3q4d * 5.25* c1 # 23891962.5 = 23.89 GHz
# g_ccg = np.sqrt(g_q3c3d*g_q4c3d) * 2.85*c1 # 149.86 MHz

"""
20250514 - Inconsistent gs throughout the fit progress gives implication that
the gs can made smaller. Also, clipping of the model works.
"""



idxa, idxb = 69, 797
idxa, idxb = 80, 797
idxa, idxb = 80, 787
idxa, idxb = 80, 787 # assume no zz_g but use this regime to fit g_qq and g_qc
idxa, idxb = 69, 797 # good enough
idxa, idxb = 64, 802 # good enough
idxa, idxb = 68, 795 # good enough, freq_fit_cbrm(idxa)=435266, freq_fit_cbrm(idxb)=4.52e6 # fit fails
idxa, idxb = 80, 787 # normalcy
# idxa, idxb = 0, 1001

# model and guess
# x_test = x_q4q3c3z_cbr[idxa:idxb]
# y_test1 = freq_fit_cbr[idxa:idxb] # raw data
# y_test2 = freq_fit_cbrm[idxa:idxb] # corrected

geff_sim0 = g_qq_hz(fc=fc3_tune_brm, g_qjqn_hz=g_qqg, g_qjcj_hz=g_ccg, 
                    zz_hz=zz_g, f1_hz=fq3_ofs_br, det_hz=detune, max_amp=400E6, 
                    tol=1E4, gate='cz')
geff_sim0 = np.abs(geff_sim0)

def g_cemp(v, a, b, c, v0, f1):
    """
    20250514 - analysis
    we determine coupler strength by an approximation done for the coupling
    strength. The idea is to determine the voltages to which g_coup = 0.
    This determines empirical bias points.
    
    Result => not capturing the effects as the lobes saturate

    """
    return a - b*np.power(v-v0,2) - c*np.power(v-v0,4)


"""---20250514 - use univariate spline to fit data and determine signs----"""
# sp = UnivariateSpline(x_q4q3c3z_cbr[idxa:idxb], freq_fit_cbrm[idxa:idxb], s=0.5) # increasing x
# geff_sp = sp(x_q4q3c3z_cbr[idxa:idxb])
"""Assessment, not robust enough"""

# z = zscore(freq_fit_cbrm[idxa:idxb]) 
# mask = np.abs(z) < 3  # Keep only values within 3 standard deviations
# x_filtered = x_q4q3c3z_cbr[idxa:idxb][mask]
# y_filtered = freq_fit_cbrm[idxa:idxb][mask]

# spline = UnivariateSpline(x_filtered, y_filtered , s=1) # increasing x
# geff_sp = spline(x_q4q3c3z_cbr[idxa:idxb])

"""Assessment Z-score not enough - hence spline is not working. Alternative,
method would be to do iterative lmfit, on small regions prior to big regions.
Make zz_g=0, f1, f2, max_amp = fixed. Find g_qq and g_qc. Use this renewed
model to correct the figure and then do a second, precise fit."""

# end of determining presign

"""Create model for fitting data, variable => A_conv, g_sqrt, g0. 3D problem
Goal. To minimize periods using the P_11 model
"""
const1 = 350E6
const2 = const1 -50E6
gemp_test = g_cemp(v=x_q4q3c3z_cbr[idxa:idxb], a=20.5E6, b=const1, c=const2, 
                   v0=dict_g['v0'][0], f1=fq3_ofs_br)

""" based on gemp_test, parabolic bias does not capture the curvature of the
g0 data"""

""" 20250515 - simplify analysis by assuming det=0, from linear regression"""
g_qqg1 = 25090770 # 25 MHz, too high from designed
g_ccg1 = 154714911 # 154 MHz, too high from designed
geff_sim1 = g_qq_hz(fc=fc3_tune_brm, g_qjqn_hz=g_qqg1, g_qjcj_hz=g_ccg1, 
                    zz_hz=0, f1_hz=fq3_ofs_br, det_hz=0, max_amp=400E6, 
                    tol=1E4, gate='cz')
geff_sim1 = np.abs(geff_sim1)

# data between linear regressions
ax0t.plot(x_q4q3c3z_cbr[idxa:idxb], freq_fit_cbrm[idxa:idxb], 'k.', label='cbrm,') 
ax0t.plot(x_q4q3c3z_cbr[idxa:idxb], geff_sim0[idxa:idxb], 'm--', label='sim, cbrm')
ax0t.plot(fit_g[0], fit_g[1], 'r-.', label='cbr')
ax0t.plot(x_q4q3c3z_cbr[idxa:idxb], geff_sim1[idxa:idxb], 'c-', 
          label='sim1, cbrm') #20250515

ax0t.set_ylim(0,400E6) # after 20250420

ax0t.legend(loc='best')
plt.show()

"""--------------Testing fit functions for the coupling strength plot-------"""
print('\n')
print('dict_coup')
# there is a slight offset from -0.150 for fc3 to -0.1483
dict_fc = {'A_conv': A_conv_c3, 
           'v0': dict_g['v0'][0], 
           'f_max': dict_fc3s['f_max'][0][0], 
           'fact': dict_fc3s['fact'][0][0], 
           'f_Ec': dict_fc3s['f_Ec'][0][0]}

dict_gc = guess_g_qcq_hz1(xdata=x_q4q3c3z_cbr[idxa:idxb], 
                          ydata=freq_fit_cbrm[idxa:idxb], 
                          f1=fq3_ofs_br, gate='cz', dict_fc=dict_fc, 
                          show=['N','Y'], num_fitfrac=30)
print(dict_gc)

"""
guess_params
{'gic_hz': 154473160.9331757, 'K': 6.1654319915999425, 
 'gqq_hz': 25054718.167946186, 'f1_hz': 4719293199.340988}
guess_g_qcq_hz1 is meant to overestimage g_qq and g_ic due to optimum condition
that f1=f2. But one can use this as an order-guess and a better guess for 
"""

print('\n')

lst_g_fit, dict_g_fit = lm_min_gqcq_hz1(xdata=x_q4q3c3z_cbr[idxa:idxb], 
                                        ydata=freq_fit_cbrm[idxa:idxb], 
                                        guess=[180E6,  
                                               6, 
                                               36E6,
                                               fq3_ofs_br], 
                                        gate='cz', tol=1E4, 
                                        max_amp=200E6, dict_fc=dict_fc, 
                                        show=['N','Y'],
                                        method='Nelder',
                                        bool_params=[True,True,False,False])

"""
Nonlinear fit: - idxa:idxb = 80:787 => different from linear regime
{'gic_hz': [146820318.48385343, 381693.9552950054], 
 'K': [6.1391955560288745, 0.0031811443329942935], 
 'gqq_hz': [23915237.288649563, 63396.23579393349], 
 'f1_hz': [4719293199.340988, 0], 'device': 'qubit'}

this initial fit can aid in selecting xdata whose values are above 0, 
and do sign corrections.
"""
# idxa, idxb = 69, 797 # fit failes hard
# lst_g_fit, dict_g_fit = lm_min_gqcq_hz1(xdata=x_q4q3c3z_cbr[idxa:idxb], 
#                                         ydata=freq_fit_cbrm[idxa:idxb], 
#                                         guess=[146820318.48385343,  
#                                                6.1391955560288745, 
#                                                36E6,
#                                                fq3_ofs_br], 
#                                         gate='cz', tol=1E4, 
#                                         max_amp=450E6, dict_fc=dict_fc, 
#                                         show=['Y','Y'],
#                                         method='Nelder',
#                                         bool_params=[True,False,False,False])
"""Fit fails."""

"""
Fitting for zz_g (residual coupling strength), without coupler, 1 Mhz, 
with coupler < 100 kHz. In our case, zz-interaction ranges from 1 MHz to 5 MHz.

"""

"""20250516 - Experiment with dict_g for flipping data frequencies"""
idxa, idxb = 0,1000
dict_g_fit2 = {'gic_hz': dict_g_fit['gic_hz'][0],
               'K': dict_g_fit['K'][0],
               'gqq_hz': dict_g_fit['gqq_hz'][0],
               'f1_hz': dict_g_fit['f1_hz'][0]}
freq_fit_mod = flip_sign_mod(xdata=x_q4q3c3z_cbr[idxa:idxb], 
                             ydata=freq_fit_cbrm[idxa:idxb], 
                             dict_fc=dict_fc, dict_fit=dict_g_fit2)

geff_sim2 = g_qq_hz(fc=fc3_tune_brm, g_qjqn_hz=dict_g_fit['gqq_hz'][0], 
                    g_qjcj_hz=dict_g_fit['gic_hz'][0], zz_hz=0, 
                    f1_hz=fq3_ofs_br, det_hz=0, max_amp=400E6, 
                    tol=1E4, gate='cz')
geff_sim2 = np.abs(geff_sim2)
amp1, amp2 = -22E6, 22E6
amp1, amp2 = -200E6, 400E6

plt.plot(x_q4q3c3z_cbr[idxa:idxb], freq_fit_cbrm[idxa:idxb], 'k.', label='cbrm') 
plt.plot(x_q4q3c3z_cbr[idxa:idxb], geff_sim2[idxa:idxb], 'b-.', label='sim, narrow')
plt.plot(x_q4q3c3z_cbr[idxa:idxb], freq_fit_mod, 'r.', label='flipped') 
plt.xlabel('C3 voltage')
plt.ylabel(r'$g/2\pi$ (MHz)')
plt.ylim(amp1,amp2) # after 20250511
# plt.ylim(0,400E6)
plt.show()

"""-20250516 - Flip data and experiment------------------------------"""
idxa, idxb = 68, 797 # good enough, freq_fit_cbrm(idxa)=435266, freq_fit_cbrm(idxb)=4.52e6 # fit fails
idxa, idxb = 80, 787 # data measurement
print('coup_2_with_zz and detuning')
dict_gc2 = guess_g_qcq_hz2(xdata=x_q4q3c3z_cbr[idxa:idxb], 
                           ydata=freq_fit_cbrm[idxa:idxb], 
                           f1=fq3_ofs_br, det=q4q3_cz_detune, gate='cz', dict_fc=dict_fc, 
                           show=['N','N'], num_fitfrac=30)
print(dict_gc2)
print('fin')
print('\n')


"""
coup_2_with_zz and detuning
{'gic_hz': 217568863.0685656, 'K': 10.330422271553743, 
 'gqq_hz': 21060984.47375881, 'zz_hz': 3899647.827528759, 
 'f1_hz': 4719293199.340988, 'det_hz': 9329921.550110817}


"""

# remove high peak outliers
x_fit_tst = x_q4q3c3z_cbr[idxa:idxb]
y_fit_tst = freq_fit_cbrm[idxa:idxb]
idx_fit_tst = np.where(y_fit_tst<=25E6)[0]
#print(idx_fit_tst)
x_fit_tst1 = x_fit_tst[idx_fit_tst]
y_fit_tst1 = y_fit_tst[idx_fit_tst]

# plt.plot(x_fit_tst1, y_fit_tst1)
# plt.show()
method='nelder' # causes strip
method='leastsq'

# too-many parameters
zz_g = 0E6 # converges, prior to 20250539
# zz_g = 4.52311E6 # does not converge
# zz_g = 435266 # off in correction
# zz_g = (4.52311E6 + 435226)/2, zz_l=y_cbrm[68], zz_r=y_cbrm[795]
# zz_g = 0E6
# this model could not account for the prediction, 20250523
zz_g = (435266 + 4.52311E6)/2 # no good splitting, zz_l=y_cbrm[65], zz_r=y_cbrm[795]
zz_g = 435266

# 20250530

# model fully used for calibration
lst_g1_fit, dict_g1_fit = lm_min_gqcq_hz2(xdata=x_fit_tst1, 
                                        ydata=y_fit_tst1, 
                                        guess=[180E6,  
                                                6, 
                                                36E6,
                                                fq3_ofs_br,
                                                q4q3_cz_detune,
                                                zz_g], 
                                        gate='cz', tol=1E4, 
                                        max_amp=400E6, dict_fc=dict_fc, 
                                        show=['Y','Y'],
                                        method=method,
                                        bool_params=[True,True,False,False,False,False])

print(dict_g1_fit)
"""
A_conv_c3 = fit + 0.2
zz_g = 4.5E6
Fit result within idxa, idxb = 80, 787 # data measurement
{'gic_hz': [146470524.2827176, 381618.20808775694], 
 'K': [7.566449019832972, 0.006292583517703392], 
 'gqq_hz': [19357894.819457978, 52942.61412261627], 
 'f1_hz': [4719293199.340988, 0], 
 'det_hz': [9329921.550110817, 0], 
 'zz_hz': [4523110.0, 0], 'device': 'qubit'}

A_conv_c3 = fit + 0.2
zz_g = 4.5E6

Fit result within idxa, idxb = 68, 797 # data measurement => not accurate
{'gic_hz': [138414276.7435368, 178.2358666675879], 
 'K': [7.62535051638008, 0.00902436656935466], 
 'gqq_hz': [18151857.602638453, 21482.174955194663], 
 'f1_hz': [4719293199.340988, 0], 
 'det_hz': [9329921.550110817, 0], 
 'zz_hz': [4523110.0, 0], 'device': 'qubit'}

for z_g = 4.5E6 
{'gic_hz': [146470524.2827176, 381618.20808775694], 
 'K': [7.566449019832972, 0.006292583517703392], 
 'gqq_hz': [19357894.819457978, 52942.61412261627], '
 f1_hz': [4719293199.340988, 0], 'det_hz': [9329921.550110817, 0], 
 'zz_hz': [4523110.0, 0], 'device': 'qubit'}

for zz_g = 0
{'gic_hz': [146470524.2827078, 381618.2080956996], 
 'K': [6.133348466282857, 0.0031921409959262824], 
 'gqq_hz': [23881004.819456626, 63449.46596366194], 
 'f1_hz': [4719293199.340988, 0], 
 'det_hz': [9329921.550110817, 0], 
 'zz_hz': [0.0, 0], 'device': 'qubit'}
"""

idxa, idxb = 0, 1000
amp1, amp2 = -22E6, 22E6
# amp1, amp2 = -5E6, 400E6

#zz_l = y_trans[68]= 435266 Hz, zz_r = y_trans[795] 4.52311e+06, zz_avg = 2,479,188
zz_g = dict_g1_fit['zz_hz'][0]
# 20250523
zz_g = (435266 + 4.52311E6)/2
zz_g = 435266

# 20250530, with symmetric dataset
zz_g = (431197 + 4.61219e+06)/2
zz_g = 431197 

geff_sim2 = g_qq_hz(fc=fc3_tune_brm[idxa:idxb], g_qjqn_hz=dict_g1_fit['gqq_hz'][0], 
                    g_qjcj_hz=dict_g1_fit['gic_hz'][0], 
                    zz_hz=zz_g , 
                    f1_hz=fq3_ofs_br, det_hz=q4q3_cz_detune, max_amp=400E6, 
                    tol=1E4, gate='cz')
geff_sim2 = np.abs(geff_sim2)

dict_g_fit3 = {'gic_hz': dict_g1_fit['gic_hz'][0],
               'K': dict_g1_fit['K'][0],
               'gqq_hz': dict_g1_fit['gqq_hz'][0],
               'f1_hz': dict_g1_fit['f1_hz'][0],
               'det_hz': dict_g1_fit['det_hz'][0],
               'zz_hz': zz_g}

x_trans = x_q4q3c3z_cbr[idxa:idxb]
y_trans = freq_fit_cbrm[idxa:idxb]

# this fit needs some automation or reasonable guesses
freq_fit_mod2 = flip_sign_mod(xdata=x_trans, 
                              ydata=y_trans, 
                              dict_fc=dict_fc, 
                              dict_fit=dict_g_fit3, 
                              zz_l=y_trans[65], 
                              zz_r=y_trans[795])

# details => 
plt.plot(x_q4q3c3z_cbr[idxa:idxb], freq_fit_cbrm[idxa:idxb], 'k.', label='cbrm') 
plt.plot(x_q4q3c3z_cbr[idxa:idxb], geff_sim2[idxa:idxb], 'b-', label='sim1, with zz')
plt.plot(x_trans, freq_fit_mod2, 'r.', label='sim2, with zz')

plt.xlabel('C3 voltage')
plt.ylabel(r'$g/2\pi$ (MHz)')
plt.ylim(amp1,amp2) # after 20250511
plt.ylim(amp1,amp2)
plt.legend(loc='best')
plt.show()


"""---------Data Format for Publishable Figures-----------------------------"""
wfig=8.6
set_font_default(9)

# set primary axis, bridge
xlbl_br0 = 'C3 Flux Bias (V)' # previous
xlbl_br1 = r'Coupler Flux ($\pi \Phi_{C3} / \Phi_0$)' # normalized
ylbl_br = 'Time (ns)'
ylbl_br2 = r'$|g_{eff}$/(2$\pi$)| (MHz)'
Ilbl_br = r'$|11\rangle$ population ($\mu$V)' # next time apply this to |11> and |20>

br_xlim = (-2, 0.17)
# br_xlim = (-1.9, 0.1)
# 20250511
br_xlim = (x_q4q3c3z_cbr[idx_1], x_q4q3c3z_cbr[idx_2])

""" Just for showing, 20250424 """
br_xlim1 = volts_to_norm_flux(vpk=(-2, 0.17), 
                              A_conv=dict_fc3s['A_conv'][0][0], 
                              v0=dict_fc3s['v0'][0][0])
# with modified br_xlim2 based on observations
br_xlim2 = volts_to_norm_flux(vpk=np.array([-2, 0.17]), 
                              A_conv=dict_fc3['A_conv'][0][0], 
                              v0=-0.8675)

# 20250511 - disabled work.
idx_1 = 53 # g/2pi = 5.09189e+06
idx_1 = 0 # g/2pi = 5.09189e+06
idx_2 = 868 # g/2pi = 5.05251e+07
br_xlim2 = volts_to_norm_flux(vpk=np.array([x_q4q3c3z_cbr[idx_1], 
                                            x_q4q3c3z_cbr[idx_2]]), 
                              A_conv=A_conv_c3, 
                              v0=dict_g['v0'][0])

# 20250521 - word done conversion

""" 20250502 - using twin-axes to map $\Delta_{1}$ (affects coupler flux in czbr)
and Coupler Flux (C3) $\Delta_{1}$ and Q4 Flux (\pi \Phi_{Q4}/\Phi_{0}). 
Add twinX of br_xlim2 and """

br_ylim = (-0.05, 100)
br_ylim = (-0.05, 100)

xticks_q4flux_br = np.array([-2, -1.5, -1.0, -0.5, 0]) # Volts_pk
yticks_q4time_br = np.array([0, 100, 200, 300, 400]) # ns time scale
yticks_geff = np.array([0,25,50,75,100]) # coupling strength
# later create ticks
zticks_q4time_br = []

# set primary axis, leakage
xlbl_lk = 'Q4 Flux Bias (V)' # original
xlbl_lk = 'Q4 Flux Bias (mV)' # with tick renaming
xlbl_lk = 'Q4 Flux ($\pi \Phi_{Q4} / \Phi_0$)'
ylbl_lk = 'C3 Flux Bias (V)'  # original
ylbl_lk = 'C3 Flux Bias (mV)' # with tick renaming
ylbl_lk = xlbl_br1
Ilbl_lk = r'|11$\rangle$ population ($\mu$V)'

xticks_q3flux_lk = np.array([-20, -15, -10, -5, 0]) # milliVolts
xticks_q3flux_lk = np.array([-0.2, -0.1, 0, 0.1, 0.2]) # Volts
xticks_q3flux_lk_str = ["-200", "-100", "0", "100", "200"]
yticks_c3flux_lk = np.array([0.10, 0.125, 0.15, 0.175, 0.2]) # in volts
yticks_c3flux_lk_str = ["100", "125", "150", "175", "200"]
# later create ticks
zticks_c3time_lk = []

lk_xlim = (-0.114635, 0.138496)
lk_ylim = (0.988292, 1.08522)
# xticks_q4flux_lk = 

# set secondary axis, bridg
secax_xlbl_br = 'Norm. Flux Q3 ' + '()' + r'($\pi \Phi$/$\Phi_{0}$)'
secay_xlbl_br = 'Norm. Flux C2 ' + '()' + r'($\pi \Phi$/$\Phi_{0}$)'

# colorbar map
cmap = 'coolwarm'
# set fontsize global
fsize = 9
# set pad
pad = [0.1,0.1,2,1]

# put ticks in colbar, 20250502
# note, cbar currently for czbr currently disabled
cbar_czbr = [[0.35,0.68,0.10,0.01],'horizontal'] 
cbar_czlk = [[0.35,0.68,0.10,0.01],'horizontal']


# 20250502 - improved delivery of data


# repository for save-fig
fname_fig5 = dir_main + dir_fig5 + 'fig5_CZ_bridge_leak.png'
fname_fig5 = dir_main + dir_fig5 + 'fig5_CZ_bridge_leak_2.png'
fname_fig5 = dir_main + dir_fig5 + 'fig5_CZ_bridge_leak_3.png'
fname_fig5 = dir_main + dir_fig5 + 'fig5_CZ_bridge_leak_4.png'
#20250529
fname_fig5 = dir_main + dir_fig5 + 'fig5_CZ_bridge_leak_3_1.png' # this works.
# suggestion by CTK, combine this with some data from the simulation.
"""
requires export of current figure to another figure and add the simulation
plots to confirm making works.
"""

"""------------------------Plot figures-------------------------------------"""
fig = plt.figure(constrained_layout=True, figsize=(2.0*cm_to_inch(wfig),
                                                    1.0*cm_to_inch(wfig)))

# main text, v2, gridspec
# gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], 
#                        wspace=0.1, hspace=0.05)  # Adjusting spacing, vertical strip
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], 
                        wspace=0.1, hspace=0.05)  # Adjusting spacing, two-column page strip

# First group (ax0 (ul), ax1 (ur), ax2 (ll), ax3 (lr))
gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0.0, wspace=0.1, 
                                        subplot_spec=gs[0])

# advantage - shared x-y axis for comparison
ax0 = fig.add_subplot(gs0[0, 0]) # bridge prior to correction, with title
ax1 = fig.add_subplot(gs0[1, 0]) # bridge after correction, with title
ax2 = fig.add_subplot(gs0[2, 0]) # extracted coupling strength, no title


gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0.0, subplot_spec=gs[1])
# Third group (ax4 and ax5 stacked with different y-labels)
ax3 = fig.add_subplot(gs1[0, 0])
ax4 = fig.add_subplot(gs1[1, 0])
# ax5 = fig.add_subplot(gs1[2, 0])

# intensity axis scaling
b1 = 1E6 # conversion from V to uV (for z-axis)
# bridge measurement
b2 = 1E-6 # conversion from Hz to MHz (for y-axis)

# # #Q0 => Q3, Q1 => Q4, Q0_RO => Q3, CZ = Q0 

"""-------------Modified Figure with axes correction (flux bias / normalized to
flux quanta)------------"""
"""Shared colorbar for two meshes"""
# Set a common color range for each gridspec
vmin1 = min(I_q4q3c3z_ubrt.min(), I_q4q3c3z_cbrt.min())*b1 
vmax1 = max(I_q4q3c3z_ubrt.max(), I_q4q3c3z_cbrt.max())*b1
vmin2 = min(I_q4q3c3z_ulk.min(), I_q4q3c3z_clk.min())*b1
vmax2 = max(I_q4q3c3z_ulk.max(), I_q4q3c3z_clk.max())*b1

ax0.set_title('CZ SWAP, Uncompensated', fontsize=fsize)
mesh0 = color_map(fig=fig, ax=ax0, x_dat=norm_flux_c3_brm, y_dat=y_q4q3c3z_ubr, 
                  z_dat=I_q4q3c3z_ubrt*b1, 
                  axs_lbl=['', ylbl_br, Ilbl_br], cmap=cmap, 
                  ticks_2D=[[], yticks_q4time_br,[]], 
                  pad_dat=pad, font_size=fsize, prop_colbar=cbar_czbr, 
                  norm='N', cbar=False)
mesh0.set_clim(vmin=vmin1, vmax=vmax1)

ax0.set_xticks(ticks=xticks_q4flux_br, labels="") # # axes.ytickslabel off
ax0.set_xlim(br_xlim2)

ax1.set_title('CZ SWAP, Flux Compensated', fontsize=fsize)
mesh1 = color_map(fig=fig, ax=ax1, x_dat=norm_flux_c3_brm, y_dat=y_q4q3c3z_cbr, 
                  z_dat=I_q4q3c3z_cbrt*b1, 
                  axs_lbl=['', ylbl_br, Ilbl_br], cmap=cmap, 
                  ticks_2D=[[], yticks_q4time_br,[]], 
                  pad_dat=pad, font_size=fsize, prop_colbar=cbar_czbr, 
                  norm='N', cbar=False)
mesh1.set_clim(vmin=vmin1, vmax=vmax1)
# Add shared colorbars
cbar1 = fig.colorbar(mesh1, ax=[ax0, ax1], orientation="vertical", shrink=1)
cbar1.set_label(Ilbl_br)

ax1.set_xticks(ticks=xticks_q4flux_br, labels="") # # axes.ytickslabel off
ax1.set_xlim(br_xlim2)

# set common colorbar
ax2.plot(norm_flux_c3_brm, freq_fit_ubrm*b2, 'r-', label='Uncompensated')
ax2.plot(norm_flux_c3_brm, freq_fit_cbrm*b2, 'b--', label='Flux Compensated')
ax2.plot(norm_flux_c3_brm, geff_sim2*b2, 'g-', label='Fit') 
# ax2.set_yscale('log')
ax2.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                right = True)
ax2.set_yticks(ticks=yticks_geff)
ax2.set_xlabel(xlbl_br1)
ax2.set_ylabel(ylbl_br2)
ax2.set_xlim(br_xlim2)
ax2.set_ylim(br_ylim)
ax2.legend(loc='best', frameon=False)

ax3.set_title('CZ Leakage, Uncompensated', fontsize=fsize) # normalized to flux quanta
mesh3 = color_map(fig=fig, ax=ax3, x_dat=norm_flux_q4_lk, y_dat=norm_flux_c3_lk, 
                  z_dat=I_q4q3c3z_ulk*b1, 
                  axs_lbl=['', ylbl_lk, Ilbl_lk], 
                  cmap=cmap, ticks_2D=[[], [],[]], pad_dat=pad, 
                  font_size=fsize, prop_colbar=cbar_czlk, 
                  norm='N', cbar=False)
# ax3.set_xticks(ticks=xticks_q3flux_lk, labels="") # # axes.ytickslabel off
# ax3.set_yticks(ticks=yticks_c3flux_lk, labels=yticks_c3flux_lk_str)
# mesh3.set_clim(vmin=vmin2, vmax=vmax2)

ax4.set_title('CZ Leakage, Flux Compensated', fontsize=fsize) # normalized to flux quanta
mesh4 = color_map(fig=fig, ax=ax4, x_dat=norm_flux_q4_lk, y_dat=norm_flux_c3_lk, 
                    z_dat=I_q4q3c3z_clk*b1, 
                    axs_lbl=[xlbl_lk, ylbl_lk, Ilbl_lk], 
                    cmap=cmap, ticks_2D=[[], [],[]], 
                    pad_dat=pad, font_size=fsize, prop_colbar=cbar_czlk, 
                    norm='N', cbar=False)
# ax4.set_xticks(ticks=xticks_q3flux_lk, labels=xticks_q3flux_lk_str) # # axes.ytickslabel off
# ax4.set_yticks(ticks=yticks_c3flux_lk, labels=yticks_c3flux_lk_str) # # axes.ytickslabel off

mesh4.set_clim(vmin=vmin2, vmax=vmax2)
# Add shared colorbars
cbar2 = fig.colorbar(mesh4, ax=[ax3, ax4], orientation="vertical", shrink=1)
cbar2.set_label(Ilbl_lk)

# no 2D plot

#vmin2 
# vmax2 = min(Z3.min(), Z4.min()), max(Z3.max(), Z4.max())

# ax5.tick_params(axis = 'both', which ='both', direction='in', top=True, 
#                 right = True)
# ax5.set_xlabel(xlbl_br)
# ax5.set_ylabel(ylbl_lk)
# ax5.set_xlim(br_xlim)
# ax5.set_ylim(br_ylim)

# plt.savefig(fname=fname_fig5, dpi=300)

plt.show()

"""-20250530 - 
Save arrays used in figplot and migrate it to si-fsim for model-tuneup------
save the following 1D array
1. norm_flux_c3_brm
2. y_q4q3c3z_ubr
3. I_q4q3c3z_ubrt # it has transformation, rotation
4. y_q4q3c3z_cbr
5. I_q4q3c3z_cbrt # it has transformation, rotation
6. freq_fit_ubrm
7. freq_fit_cbrm
8. norm_flux_q4_lk
9. norm_flux_c3_lk
10. I_q4q3c3z_ulk # it has transformation, rotation
11. I_q4q3c3z_clk # it has transformation, rotation
12. geff_sim2 #f_sim
13. P_var #optional but for SI / leakage (1 -twin-axes)
"""
