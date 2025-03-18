# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:43:04 2025

Reproduce and test fit model for superconducting qubits
-> priority because of prof. Ke's paper.

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt

from ls_qbt_cplr_volts_hz import f01_hz_to_volts, f01_volts_to_hz
from ls_qbt_cplr_volts_hz import d_to_fact

from lm_utils import get_1D_plot_from_csv
from lm_qbt_cplr_volts_hz_v1 import guess_f01_volts_to_hz
from lm_qbt_cplr_volts_hz_v1 import lm_f01_volts_to_hz


"""-------- Parameters for Guo-Ping-Guo Analysis, Applied Science ----------"""
# database
dir_gpg = 'guo_ping_guo_appl_sci/'
file_gpg = dir_gpg + 'appl_sci_zbias_V_vs_freq_Hz'

fq_vs_V_gpg = get_1D_plot_from_csv(file_gpg)

# parameters

f_max_apl_sci = 5423.5E6
f_ec_apl_sci = 237E6
d_apl_sci = -0.2447
fact_apl_sci = d_to_fact(d_apl_sci) # val = 0.6068128866393508
M_apl_sci = 2.062 # rad/V
v0_apl_sci = -0.0043 # V

# modelled data
zbias_apl_sci = np.linspace(-0.3,0.3,601) # in volts
f01_apl_sci = f01_volts_to_hz(vpk=zbias_apl_sci, A_conv=M_apl_sci, 
                               v0=v0_apl_sci, f_max=f_max_apl_sci, 
                               f_Ec=f_ec_apl_sci, fact=fact_apl_sci)

"""
guess funct
    const_c:        0.50000000 +/- 0.00192654 (0.39%) (init = 0.8864848)
    cos_amplitude:  0.50000000 +/- 0.00192654 (0.39%) == '1 - const_c'
    cos_frequency:  3.90858221 +/- 0.06332564 (1.62%) (init = 10.29419)
    cos_shift:      0.01842643 +/- 5.2312e-04 (2.84%) (init = 0.06220976)
"""

# fit parameters
guess_lit = [M_apl_sci, v0_apl_sci,f_max_apl_sci, fact_apl_sci, f_ec_apl_sci] # lit
fit_fq_gpg, dict_fq_gpg= lm_f01_volts_to_hz(xdata=fq_vs_V_gpg[0], 
                                            ydata=fq_vs_V_gpg[1], 
                                            show=['Y','N'],
                                            guess=guess_lit,
                                            device='Q0_gpg',
                                            method='least_sq') # least squares is the way to go

# check guess
dict_guess = guess_f01_volts_to_hz(xdata=fq_vs_V_gpg[0], 
                                   ydata=fq_vs_V_gpg[1], 
                                   f_Ec=f_ec_apl_sci)


guess_pars = [dict_guess['A_conv'], dict_guess['v0'], dict_guess['f_max'], 
              dict_guess['fact'], f_ec_apl_sci]

fit_fq_fit, dict_fq_fit= lm_f01_volts_to_hz(xdata=fq_vs_V_gpg[0], 
                                            ydata=fq_vs_V_gpg[1], 
                                            show=['Y','N'],
                                            guess=guess_pars,
                                            bool_params = [True, True, True, True, False],
                                            device='Q0_gpg',
                                            method='least_sq')

"""
A_conv:  2.05336642 +/- 0.03809838 (1.86%) (init = 1.616697)
v0:     -0.00474403 +/- 1.1059e-04 (2.33%) (init = -0.00969564)
f_max:   5.4250e+09 +/- 393922.254 (0.01%) (init = 5.42473e+09)
fact:    0.54775925 +/- 0.06164296 (11.25%) (init = 0)
f_Ec:    2.37e+08 (fixed)
d: -0.292, 0.0514
In good agreement with literature
"""

# plot data
b1 = 1E-6
plt.plot(fq_vs_V_gpg[0], fq_vs_V_gpg[1]*b1, 'k.', label='GPG_data')
plt.plot(zbias_apl_sci, f01_apl_sci*b1, 'r:', label='GPG_model')
plt.plot(fit_fq_gpg[0], fit_fq_gpg[1]*b1, 'm-', label='lm_fit')
plt.xlabel('Z-bias (Voltage)')
plt.ylabel('Frequency (MHz)')
plt.xlim(-0.3,0.3)
plt.ylim(4850,5500)
plt.legend(loc='best')
plt.show()

"""See if selected Z-bias from fit provides useful bias mapping"""
zbias_select_1 = f01_hz_to_volts(f01=4.92521e+09, A_conv=M_apl_sci, 
                                 v0=v0_apl_sci, f_max=f_max_apl_sci, 
                                 f_Ec=f_ec_apl_sci, fact=fact_apl_sci)
print('zbias_select_1')
print(zbias_select_1)
"""
zbias_select = [-0.29999985582073274, 0.2913998558207327] => correct
f01_apl_sci = [4.92521e+09, [4.92657e+09,4.92317e+09]]
"""
zbias_select_2 = f01_hz_to_volts(f01=5423.5E6, A_conv=M_apl_sci, 
                                 v0=v0_apl_sci, f_max=f_max_apl_sci, 
                                 f_Ec=f_ec_apl_sci, fact=fact_apl_sci)
print('zbias_select_2')
print(zbias_select_2)
"""
zbias_select_2 = [-0.0043, -0.0043] => v0_apl_sci +/- volts
"""

f01_target = np.linspace(4.950, 5.400, 10)*1E9
zbias_select_3 = f01_hz_to_volts(f01=f01_target, A_conv=M_apl_sci, 
                                 v0=v0_apl_sci, f_max=f_max_apl_sci, 
                                 f_Ec=f_ec_apl_sci, fact=fact_apl_sci)
print('zbias_select_3')
print(zbias_select_3)
"""
[array([-0.29260459, -0.27706123, -0.26055085, -0.24287272, -0.22374481,
       -0.20274844, -0.17921218, -0.15192645, -0.11822883, -0.06874392]), array([0.28400459, 0.26846123, 0.25195085, 0.23427272, 0.21514481,
       0.19414844, 0.17061218, 0.14332645, 0.10962883, 0.06014392])]
# Note that this results require information
"""

"""
We added effort in improving the robustness of the fit, only to realize that
puting fact into 0 made it more stable than putting any value of d=1. gam=1.
The sinusoidal fit allws better guess, but somehow, the program got stock
to a local minum. One reason this could be the case, is simply the issue
of the published paper, which is not exact. Nevertheless, we will compare the
approaches in the later run.
"""
