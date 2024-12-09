# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:26:36 2024

Demonstration of deterministic benchmarking tools for comparison
paper - https://arxiv.org/abs/2407.09942

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_plots_qm import cm_to_inch
from lm_utils import retrieve_data_from_csv_list
from lineshapes_det_bench import dB_errors, get_db_errors
from lineshapes_det_bench import dB_fidelity, get_spam_error
from lm_det_bench import lm_dB_errors, lm_dB_infid
from lm_det_bench import steps_db_errors

"""Retrieve sets of data Data"""

dir_name = ''
fname_fig2_lst = ['Fig2_Free_1_us_vs_F', 'Fig2_XX_+_us_vs_F', 
                  'Fig2_XX-_+_us_vs_F', 'Fig2_YY_+_us_vs_F',
                  'Fig2_YY-_+_us_vs_F', 'Fig2_Y-Y_+_us_vs_F']
fname_fig3_lst = ['Fig3_YY_+_us_vs_F', 'Fig3_XX-_+_us_vs_F', 'Fig3_UR6_+_us_vs_F']
fname_fig5_lst = ['Fig5_rb_nclif_vs_F']

fig2_data = retrieve_data_from_csv_list(dir_name, fname_fig2_lst)
fig3_data = retrieve_data_from_csv_list(dir_name, fname_fig3_lst)
fig5_data = retrieve_data_from_csv_list(dir_name, fname_fig5_lst)

"""Extract T1 time"""
T1_fig2 = 23360 # ns
T2r_fig2 = 44130 # ns
T_coh1 = 8000
T_coh2 = 10000
T_coh3 = 0
T_coh4 = T_coh3
del_theta = 0.398 # degrees, literature (wrong value), corrected due to definition
#del_theta = 1.75 # degrees, fit by eye, empirical fit
del_phi = 0.426 # degrees, correct value

# if swapping frequencies
# del_theta = 0.426  #f_theta = 6.7234848
# del_phi = 0.398 #f_phi = 1.2563E-6

"""We hypothesize that the inputted values for theta and phi got swapped out
because of the difference in frequency."""

a_bar = 0.22
a_bar = 0.7 # moduled data

tg = 88 #ns 
t_arr = np.linspace(0, 2*tg*500, 500+1) #strict definition of tg
n_arr = np.linspace(0, 500, 500+1)

# modelled
f_theta = np.deg2rad(del_theta)/(2*np.pi*2*tg) 
#6.2816E-6 - magnitude does not make sense => assigned to YY
f_phi = np.deg2rad(del_phi)/(2*np.pi*tg) 
#1.3447E-5 - magnitude does not make sense given frequency => asigned to XXbar
a_yy = 0.05 #+/- 0.01, yy
a_yybar = 0.22 #+/- 0.01, ybary or yybar

"""Check if model scites what it claims"""

test_free = dB_errors(tn=t_arr, f_e=0, T_d=T1_fig2, a=-1) #a = -1 for decay
test_xx = dB_errors(tn=t_arr, f_e=0, T_d=T2r_fig2, a=0)
test_yy = dB_errors(tn=t_arr, f_e=f_theta, T_d=T1_fig2+T_coh2, a=-0.05)
test_xxbar = dB_errors(tn=t_arr, f_e=f_phi, T_d=T1_fig2+T_coh1, a=-0.05)

# solved via numerics of Figure 1-4
test_ybary = dB_errors(tn=t_arr, f_e=0, T_d=T2r_fig2+T_coh3, a=a_bar)
test_yybar = dB_errors(tn=t_arr, f_e=0, T_d=T2r_fig2+T_coh4, a=-a_bar)

# get randomized benchmarking values
test_ybary_rb = dB_fidelity(n=n_arr, A=0.2527, B=0.7739, r_clif=2.72E-3, num_qubits=1)

"""for fitting purposes,
sequence of guesses
1. 
get the frequency using cosine (to set omega without theta)
2. 
get the decay time
3. 
use decay time and frequency to get a
"""

"""Plot and test guess functions - Fig2"""
# wfig=8.6
# fig = plt.figure(constrained_layout=True, figsize=(1*cm_to_inch(wfig),
#                                                     1*cm_to_inch(wfig)))
# spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
# ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model

# ax0.set_title('Fig. 2: Manual fit')
# # first four info - extraction
# # T1
# ax0.plot(fig2_data[0][0], fig2_data[0][1], 'k.', label=r'Free;$|0>$')
# ax0.plot(t_arr/1000, test_free, 'm:', label='model, Free;$|0>$')
# # T2
# ax0.plot(fig2_data[1][0], fig2_data[1][1], 'b.', label=r'$XX$;$|+>$')
# ax0.plot(t_arr/1000, test_xx, 'm:', label='model, $XX$;$|+>$')
# # rot_error
# ax0.plot(fig2_data[3][0], fig2_data[3][1], 'g.', label=r'$YY$;$|+>$')
# ax0.plot(t_arr/1000, test_yy, 'b:', label='model, $YY$;$|+>$')
# # phase_error
# ax0.plot(fig2_data[2][0], fig2_data[2][1], 'c.', label=r'$X\bar{X}$;$|+>$')
# ax0.plot(t_arr/1000, test_xxbar, 'b:', label='model, $X\bar{X}$;$|+>$')


# # test model - test model, not corresponding to model, cannot be explained by empirical
# ax0.plot(fig2_data[4][0], fig2_data[4][1], 'y.', label=r'$Y\bar{Y}$;$|+>$')
# ax0.plot(t_arr/1000, test_ybary, 'g:', label='model, $YY$;$|+>$')

# ax0.plot(fig2_data[5][0], fig2_data[5][1], 'm.', label=r'$\bar{Y}Y$;$|+>$')
# ax0.plot(t_arr/1000, test_yybar, 'c:', label='model, $YY$;$|+>$')

# ax0.set_xlabel(r'Evolution time ($\mu$s)')
# ax0.set_ylabel(r'Fidelity')
# ax0.set_xlim(-5, 90)
# ax0.set_ylim(-0.05, 1.05)
# ax0.legend(loc='best', ncols=2, fontsize='small')

# plt.show()

# test for RB and SPAM Errors
# wfig=8.6
# fig = plt.figure(constrained_layout=True, figsize=(1*cm_to_inch(wfig),
#                                                     1*cm_to_inch(wfig)))
# spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
# ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model
# ax0.plot(fig2_data[5][0]*1000/(2*tg), fig2_data[5][1], 'm.', label=r'$\bar{Y}Y$;$|+>$')
# ax0.plot(n_arr, test_ybary_rb, 'g--', label='rb, $YY$;$|+>$')
# ax0.set_ylim(-0.05, 1.05)
# ax0.set_xlabel(r'Number of Cliffords')
# ax0.set_ylabel(r'Fidelity')
# ax0.legend(loc='best', ncols=2, fontsize='x-small')

# plt.show()

"""--------------Corollary - Randomized Benchmarking, and SPAM Errors-------"""
fit_fig5, dict_fig5 = lm_dB_infid(xdata=fig5_data[0][0], 
                                  ydata=fig5_data[0][1], show=['Y','Y'])
"""
[[Model]]
    Model(dB_fidelity)
[[Fit Statistics]]
    # fitting method   = least_squares
    # function evals   = 20
    # data points      = 71
    # variables        = 3
    chi-square         = 0.00134157
    reduced chi-square = 1.9729e-05
    Akaike info crit   = -766.238307
    Bayesian info crit = -759.450267
    R-squared          = 0.99851493
[[Variables]]
    A:           0.42807433 +/- 0.00201415 (0.47%) (init = 0.4591913)
    B:           0.55348377 +/- 0.00144696 (0.26%) (init = 0.559043)
    r_clif:      0.00261748 +/- 3.3303e-05 (1.27%) (init = 0.002935059)
    num_qubits:  1 (fixed)
    spam_F:      0.98155810 +/- 0.00212982 (0.22%) == 'A+B'
[[Correlations]] (unreported correlations are < 0.100)
    C(B, r_clif) = +0.8534
    C(A, B)      = -0.2770
    C(A, r_clif) = +0.1098
"""
# fit rb = 0.2617 +/- 0.003%, published rb = 0.262 +/-  0.003%

"""----------------------Test of Fitting Methods----------------------------"""
e1 = 1000 # us to ns
# get T1
fit_fig2_0, dict_fit_fig2_0 = lm_dB_errors(xdata=fig2_data[0][0]*e1, 
                                            ydata=fig2_data[0][1], show=['N','Y'],
                                            bool_params = [False, True, False],
                                            guess = [0, 1E-5, -1]) # need manual control = f_e=0
# fit T1 = 23301 +/- 196 ns, published T1 = 23.36 +/- 0.40 us

# get T2
fit_fig2_1, dict_fit_fig2_1 = lm_dB_errors(xdata=fig2_data[1][0]*e1, 
                                            ydata=fig2_data[1][1], show=['N','Y'],
                                            bool_params = [False, True, False],
                                            guess = [0, 1E-5, 0]) # need manual control = f_e=0
# fit T2 = 44994 +/- 681 ns, published T2 = 44.13 +/- 2.49 us

# get rot_error. YY
fit_fig2_2, dict_fit_fig2_2 = lm_dB_errors(xdata=fig2_data[3][0]*e1, 
                                            ydata=fig2_data[3][1], show=['N','Y']) # need conversion from f_e to err_rot
# f_e=1.3616e-5, a=-0.02612 +/- 0.0064, f_theta = 6.28156E-6, T_d= 32047.2747 ns

del_rot = get_db_errors(f=dict_fit_fig2_2['f_e'][0], tg=tg, err_type='rot') 
#fit=0.3991, published rot_error=0.398

# get phase_error, XXbar

fit_fig2_3, dict_fit_fig2_3 = lm_dB_errors(xdata=fig2_data[2][0]*e1, 
                                            ydata=fig2_data[2][1], show=['N','Y']) # need conversion from f_e to err_phase
# f_e=6.2983e-6, a=-0.0514 +/- 0.0056, f_phi = 1.34469E-5, Td= 29525.4420 ns

del_pi = get_db_errors(f=dict_fit_fig2_3['f_e'][0], tg=tg, err_type='phase') #0.4313, ans=0.426
# fit=0.4313, published phase_error=0.4313

"""Report and compare - 20241205 - we have right formula but wrong result. 
The mistake comes from T1. Origin comes from T1.
"""
 
print(dict_fit_fig2_2['f_e'][0]) #1.3616E-5, f_theta_paper=6.281E-6
print(dict_fit_fig2_3['f_e'][0]) #6.298E-6, f_phi_paper=1.3447E-5
print('Rot error-fit = {:.4f} deg'.format(del_rot))
print('Rot error-fig2 = {:.4f} deg'.format(np.rad2deg(f_theta*(2*np.pi*2*tg))))
print('\n')
print('Phase error-fit = {:.4f} deg'.format(del_pi))
print('Phase error-fig2 = {:.4f} deg'.format(np.rad2deg(f_phi*2*np.pi*tg)))

"""
6.2982929550793605e-06
1.361551858350183e-05
Rot error-fit = 0.3991 deg
Rot error-fig2 = 0.3980 deg
Phase error-fit = 0.4313 deg
Phase error-fig2 = 0.4260 deg
"""

"""------------Final - streamlined data analysis of Fig2--------------------"""
fig2y_list, fig2_dict_dB = steps_db_errors(xdata_lst=[fig2_data[0][0]*e1, fig2_data[1][0]*e1,
                                                      fig2_data[3][0]*e1, fig2_data[2][0]*e1], 
                                            ydata_lst=[fig2_data[0][1], fig2_data[1][1],
                                                      fig2_data[3][1], fig2_data[2][1]], 
                                            tg=tg, show=['Y','Y'])

"""------Corollary--Assess a for fig2_4 and fig2_5--------------------------"""
"""Constant T_D* from YY - assuming same lifetime"""
# fit_fig2_4, dict_fit_fig2_4 = lm_dB_errors(xdata=fig2_data[4][0]*e1, 
#                                             ydata=fig2_data[4][1], show=['Y','Y'],
#                                             bool_params=[False,True,False],
#                                             guess=[0,dict_fit_fig2_2['T_d'][0],-0.22]) # need conversion from f_e to err_rot

"""
get |a|=0.22 - paper. YbarY, free fit, bool_params = [True, True, True]
f_e:  1.1771e-09 +/- 5.4947e-07 (46680.52%) (init = 1.113269e-05)
T_d:  33048.4346 +/- 1187.23778 (3.59%) (init = 44032.14)
a:   -0.56050005 +/- 0.02315222 (4.13%) (init = -0.01095243)
quality of fit=good

For TD=32047 ns (obtained from YY) and f_e = 0, vary a, bool_params = [False,False,True]
f_e:  0 (fixed)
T_d:  32047.17 (fixed)
a:   -0.54243614 +/- 0.00841999 (1.55%) (init = -0.01095243) - symmetric to YbarY
quality of fit=good

For f_e = 0, |a|=0.22, vary TD, bool_params = [False,True,False] (based on paper recommendations)
f_e:  0 (fixed)
T_d:  18865.3586 +/- 1129.31039 (5.99%) (init = 44032.14)
a:   -0.22 (fixed)
quality of fit=bad. => need to confirm with Dr. Jyh-Yang Wang if the |a|=0.22
for qubit temperature and compare with single-shot measurements.
"""

# fit_fig2_5, dict_fit_fig2_5 = lm_dB_errors(xdata=fig2_data[5][0]*e1, 
#                                             ydata=fig2_data[5][1], show=['Y','Y'],
#                                             bool_params=[False,True,False],
#                                             guess=[0,dict_fit_fig2_2['T_d'][0],0.22]) # need conversion from f_e to err_rot

"""
get |a|=0.22. YYbar, free fit, bool_params = [True, True, True]
f_e:  2.6333e-09 +/- 1.1789e-06 (44770.08%) (init = 1.115304e-05)
T_d:  42074.7477 +/- 3884.76114 (9.23%) (init = 43951.8)
a:    0.52166613 +/- 0.02136576 (4.10%) (init = 0.7235479)
quality of fit=good

For TD=32047 ns (obtained from YY) and f_e = 0, vary a, bool_params = [False,False,True]
f_e:  0 (fixed)
T_d:  32047.17 (fixed)
a:    0.57350194 +/- 0.00582285 (1.02%) (init = 0.7235479) - symmetric to YYbar
quality of fit=good

For f_e = 0, |a|=0.22, vary TD, bool_params = [False,True,False] (based on paper recommendations)
f_e:  0 (fixed)
T_d:  95282.3299 +/- 2363.10908 (2.48%) (init = 43951.8)
a:    0.22 (fixed)
quality of fit=bad. => need to confirm with Dr. Wang if the asymmetry of gates are good.
"""

# fit_fig2_4e, dict_fit_fig2_4e = lm_dB_infid(xdata=fig2_data[4][0]*e1/(2*tg), 
#                                             ydata=fig2_data[4][1], show=['Y','Y'],
#                                             bool_params=[True, True, True, False],
#                                             guess=[0.25, 0.77, 1E-3, 1]) # need conversion from f_e to err_rot
# spam_fig_2_4e = get_spam_error(A=dict_fit_fig2_4e['A'][0], B=dict_fit_fig2_4e['B'][0])
# print('SPAM Error={:.3e}'.format(spam_fig_2_4e))

"""
[[Model]]
    Model(dB_fidelity)
[[Fit Statistics]]
    # fitting method   = least_squares
    # function evals   = 24
    # data points      = 51
    # variables        = 3
    chi-square         = 0.02170382
    reduced chi-square = 4.5216e-04
    Akaike info crit   = -389.866714
    Bayesian info crit = -384.071237
    R-squared          = 0.98974569
[[Variables]]
    A:           0.77610942 +/- 0.01260682 (1.62%) (init = 0.8156708)
    B:           0.21290459 +/- 0.01382272 (6.49%) (init = 0.242028)
    r_clif:      0.00256504 +/- 1.2876e-04 (5.02%) (init = 0.003086537)
    num_qubits:  1 (fixed)
    spam_F:      0.98901401 +/- 0.01083244 (1.10%) == 'A+B'
[[Correlations]] (unreported correlations are < 0.100)
    C(B, r_clif) = +0.9294
    C(A, B)      = -0.6676
    C(A, r_clif) = -0.4266
    
SPAM_err = 1.099% +/- 1.083%
RB_error = 0.257 +/- 0.013%
Reported r_clif by RB benchmarking: 0.265 +/- 0.003%
"""

# fit_fig2_5f, dict_fit_fig2_5f = lm_dB_infid(xdata=fig2_data[5][0]*e1/(2*tg), 
#                                             ydata=fig2_data[5][1], show=['Y','Y'],
#                                             bool_params=[True, True, True, False],
#                                             guess=[0.25, 0.77, 1E-3, 1]) # need conversion from f_e to err_rot
# spam_fig_2_5f = get_spam_error(A=dict_fit_fig2_5f['A'][0], B=dict_fit_fig2_5f['B'][0])
# print('SPAM Error={:.3e}'.format(spam_fig_2_5f))

"""
[[Model]]
    Model(dB_fidelity)
[[Fit Statistics]]
    # fitting method   = least_squares
    # function evals   = 24
    # data points      = 51
    # variables        = 3
    chi-square         = 0.00888464
    reduced chi-square = 1.8510e-04
    Akaike info crit   = -435.418101
    Bayesian info crit = -429.622624
    R-squared          = 0.95469994
[[Variables]]
    A:           0.23913567 +/- 0.01005452 (4.20%) (init = 0.2527605)
    B:           0.76332971 +/- 0.01188737 (1.56%) (init = 0.773917)
    r_clif:      0.00215662 +/- 2.6507e-04 (12.29%) (init = 0.002720935)
    num_qubits:  1 (fixed)
    spam_F:      1.00246538 +/- 0.00667855 (0.67%) == 'A+B'
[[Correlations]] (unreported correlations are < 0.100)
    C(B, r_clif) = +0.9518
    C(A, B)      = -0.8275
    C(A, r_clif) = -0.6639

SPAM_err=-2.465e-03 = 0.2465 +/- 0.67% => not true, overcalibration
r_clif = 0.216 +/- 0.027%
Reported r_clif by RB = 0.265 +/- 0.003%
"""

"""-----------------------------Fig3_2 - confirmation of frequencies--------"""
del_rot_3d=0.995
del_phi_3d=0.90

f_theta_3d= np.deg2rad(del_rot_3d)/(2*np.pi*2*tg) 
f_phi_3d= np.deg2rad(del_phi_3d)/(2*np.pi*tg) 

""" Get Rot and Phase Error for fig3d"""
# get rot_error. YY
fit_fig3d_0, dict_fit_fig3d_0 = lm_dB_errors(xdata=fig3_data[0][0]*e1, 
                                             ydata=fig3_data[0][1], show=['Y','Y']) # need conversion from f_e to err_rot
# f_e=1.5755e-5, a=-0.04463 +/- 0.0090, f_theta = 1.5775E-5, T_d=28489.2 ns
del_rot_fig3d = get_db_errors(f=dict_fit_fig3d_0['f_e'][0], tg=tg, err_type='rot') #0.9983 deg, ans=0.9950 deg

# get phase_error, XXbar
fit_fig3d_1, dict_fit_fig3d_1 = lm_dB_errors(xdata=fig3_data[1][0]*e1, 
                                             ydata=fig3_data[1][1], show=['Y','Y']) # need conversion from f_e to err_phase
# f_e=2.4785e-4, a=0.0591 +/- 0.0074, f_phi = 2.4785E-4, Td=29806.2 ns
del_pi_fig3d = get_db_errors(f=dict_fit_fig3d_1['f_e'][0], tg=tg, err_type='phase') #0.4313, ans=0.9000 deg

print(dict_fit_fig3d_0['f_e'][0]) #1.5755E-5
print(dict_fit_fig3d_1['f_e'][0]) #2.8523E-5
print('Rot error-fit = {:.4f} deg'.format(del_rot_fig3d))
print('Rot error-fig3d_0 = {:.4f} deg'.format(np.rad2deg(f_theta_3d*(2*np.pi*2*tg))))
print('\n')
print('Phase error-fit = {:.4f} deg'.format(del_pi_fig3d))
print('Phase error-fig3d_1 = {:.4f} deg'.format(np.rad2deg(f_phi_3d*2*np.pi*tg)))

"""
1.5755227668013264e-05
2.8523281265089654e-05
Rot error-fit = 0.9983 deg
Rot error-paper = 0.9950 deg

Phase error-fit = 0.9036 deg
Phase error-paper = 0.9000 deg

Fitting model is right, if least_squares or levenberg marquardt is used.
Sometimes, result is unstable if differential evolution algorithm is used.
"""

"""Get RB from UR6 for fig3d"""
fit_fig3d_2, dict_fit_fig3d_2 = lm_dB_infid(xdata=fig3_data[2][0]*e1/(2*tg), 
                                            ydata=fig3_data[2][1], show=['Y','Y'],
                                            bool_params=[True, True, True, False],
                                            guess=[0.25, 0.77, 1E-3, 1]) # need conversion from f_e to err_rot
spam_fig_3d_f = get_spam_error(A=dict_fit_fig3d_2['A'][0], B=dict_fit_fig3d_2['B'][0])
print('SPAM Error={:.3e}'.format(spam_fig_3d_f))

"""
[[Model]]
    Model(dB_fidelity)
[[Fit Statistics]]
    # fitting method   = least_squares
    # function evals   = 20
    # data points      = 17
    # variables        = 3
    chi-square         = 0.00690815
    reduced chi-square = 4.9344e-04
    Akaike info crit   = -126.740523
    Bayesian info crit = -124.240883
    R-squared          = 0.97369619
[[Variables]]
    A:           0.45977794 +/- 0.02813696 (6.12%) (init = 0.4408829)
    B:           0.54055262 +/- 0.03193825 (5.91%) (init = 0.563019)
    r_clif:      0.00223874 +/- 3.7853e-04 (16.91%) (init = 0.002509257)
    num_qubits:  1 (fixed)
    spam_F:      1.00033056 +/- 0.01674799 (1.67%) == 'A+B'
[[Correlations]] (unreported correlations are < 0.100)
    C(B, r_clif) = +0.9490
    C(A, B)      = -0.8520
    C(A, r_clif) = -0.6962

Fit via RB
SPAM Error=-3.306e-04 => not true, overcalibration
r_clif = 0.224 +/- 0.038% 
Reported gate errors by RB: r_clif = 0.265 +/- 0.013%
"""

"""---------------------Plot and test functions, Fig3d----------------------"""

wfig=8.6
fig = plt.figure(constrained_layout=True, figsize=(1*cm_to_inch(wfig),
                                                    1*cm_to_inch(wfig)))
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
ax1 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model

# data
# rot error
ax1.plot(fig3_data[0][0], fig3_data[0][1], 'g.', label=r'data, YY')
# phase error
ax1.plot(fig3_data[1][0], fig3_data[1][1], 'y.', label=r'data, $X\bar{X}$')
# UR6. estimate r-clifford
ax1.plot(fig3_data[2][0], fig3_data[2][1], 'm.', label=r'data, UR6')

# fitting model
ax1.plot(fit_fig3d_0[0]/e1, fit_fig3d_0[1], 'g-', label=r'fit, YY')
ax1.plot(fit_fig3d_1[0]/e1, fit_fig3d_1[1], 'y-', label=r'fit, $X\bar{X}$')
# normalize X to adopt m clifford
ax1.plot(fit_fig3d_2[0]*(2*tg)/e1, fit_fig3d_2[1], 'm-', label=r'fit, UR6')

ax1.set_xlabel(r'Evolution time ($\mu$s)')
ax1.set_ylabel(r'Fidelity')
ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='best', ncols=2, fontsize='x-small')

plt.show()




"""
Conclusion - Deterministic benchmarking of rot_err and phase_err are right
Estimation of SPAM Fidelity and r_clif using RB made, not necessarily the same
as results made with RB. 

Comments
1. Debugging shows that it was a perceived order problem, where the order of
the plots thought that they represent sequencially rotation or phase errors.
2. This paper requires simulator that runs hand-in-hand for debugging. ->
digital twin qutip version of over-rotation / under-rotations.
"""

