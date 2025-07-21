"""
2024-12-26 -> Make analysis of data for efficient analysis without rescaling
Keypoint here also is to learn the dataset acquired by QM for me to readapt
the analysis in an 1) effective and efficient manner.

Comment of Yiuen, Junyi's code 
1. Fit cannot be further improved due to imperfect readout preparation. 
Method in steps are all set to varying. -> only improvement for this data 
acquisition lies with improving readout data. -> no issue with default setting
with steps (which is f_d=True, Td=True, a=True)
2. Suggestion: compress steps_dB_error into 1D dictionaries of values.

3. 

4. Transformation from state to fidelity requires non-adjusted rescaling
5. Det-bench must be converted.
6. Illustrate approximate plots

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import time
import xarray as xr

from data_plots_qm import cm_to_inch     
from lm_det_bench import steps_db_errors, steps_dB_errors_v2
from xarray import open_dataset
#dataset = open_dataset(r"D:\HW\Quela\QM\QM_data\5Q4C_20241016_2_AS1608\5Q4C_20241016_2_AS1608\Using_result_20241206\q4\20241219_171357_SQDB_shift_one_param\SQDB_shift_one_param.nc")
fname = r"C:\Users\Mai\Documents\GitHub\analyze_sqc\20241219_171357_SQDB_shift_one_param\SQDB_shift_one_param.nc"
dataset = open_dataset(fname)
tg = 40 #ns 
param_name = 'amp'

"""
Current code: Time taken: 1.070e+01 seconds
Goal: Lower benchmark time without compromising accuracy
"""

start = time.time()    
for ro_name, datas in dataset.data_vars.items():
    # retain ro_name, and datas
    xdata = datas.coords["repeat_time"].values
    #print(ro_name) #q4_ro
    #print(datas)
    """
    <xarray.DataArray 'q4_ro' (amp: 15, mixer: 1, sequence: 4, repeat_time: 800)>
    [48000 values with dtype=float64]
    Coordinates:
      * mixer        (mixer) object 'val'
      * sequence     (sequence) object 'T1' 'X X(T2 echo)' ... 'Y -Y (phase error)'
      * repeat_time  (repeat_time) int32 0 1 2 3 4 5 6 ... 794 795 796 797 798 799
      * amp          (amp) float64 0.06228 0.06328 0.06428 ... 0.07528 0.07628
    """
    
    ydata = []
    y_lists = []
    T1s = []
    T1s_std = []
    T2s = []
    T2s_std = []
    sig_thetas = []
    sig_thetas_std = []
    sig_phis = []
    sig_phis_std = []
    
    
    param_len = len(datas.coords[param_name])
    for i in range(param_len):
        ydata.append([])
        for data in datas.values[i][0]:
            ydata[i].append(data)
            
        # list of list of 4-1D numpy arrays, experiment of T1, T2_r, XX, Y-Y
        
        # can be unpacked
        
        # Inverse the state into fidelity
        for j in range(len(ydata[i])):
            # more compact way of analyzing for datasets
            ydata[i][j] = 1- ydata[i][j]
            # compact but not intuitive
            
        # Rescale exp2, exp3 and exp4,
        for j in range(1,len(ydata[i])):
            
            arr_max = ydata[i][j].max()
            arr_min = ydata[i][j].min()
            # ydata[i][j] = (ydata[i][j]-arr_min)/(arr_max - arr_min) # Max=1, min=0
            ydata[i][j] = ydata[i][j]+1-arr_max # Max=1, only shift

        # fit method - steps_dB_errors not performant on calibration, assumes ideal analysis
        
        y_list, dict_dB = steps_dB_errors_v2(xdata_lst=[xdata*tg*2, xdata*tg*2,
                                                        xdata*tg*2, xdata*tg*2], 
                                             ydata_lst=[ydata[i][0], ydata[i][1],
                                                        ydata[i][2], ydata[i][3]], 
                                             tg=tg, show=["N","N"])
        
        """
        Due to imperfect readout, we see that the oscillation for the model
        could not fit to the datasets well. Hence our frequency oscillations
        got some imperfection in the fit. Fit frequency is very much underestimated
        Observation 1
        1)
        """
        
        y_lists.append(y_list)
        T1s.append(dict_dB['T1_ns'][0])
        T1s_std.append(dict_dB['T1_ns'][1])
        T2s.append(dict_dB['T2r_ns'][0])
        T2s_std.append(dict_dB['T2r_ns'][1])
        sig_thetas.append(dict_dB['sig_theta_deg'][0])
        sig_thetas_std.append(dict_dB['sig_theta_deg'][1])
        sig_phis.append(dict_dB['sig_phi_deg'][0])
        sig_phis_std.append(dict_dB['sig_phi_deg'][1])
        """
        steps in data analysis, 40 ns, 1000 shots, 14-15 minutes.
        Whole study, 15 parameters of amp => 3.5 - 3.75 hours.
        
        Requesting autofit to contain these info in a systematic manner.
        """

end = time.time()
elapsed = end - start
print(f'Time taken: {elapsed:.3e} seconds')

"""
q4_ro    (amp, mixer, sequence, repeat_time) float64 0.792 0.811 ... 0.642)
"""


wfig=8.6
fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                    2*cm_to_inch(wfig)))
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model
ax0.errorbar(datas.coords[param_name], T1s, yerr=T1s_std, ls='solid', 
             color='k', fmt="s", label=r'T1(ns)')
ax0.errorbar(datas.coords[param_name], T2s, yerr=T2s_std, ls='solid', color='b', 
             fmt="s", label=r'T2(ns)')
ax1 = ax0.twinx()
ax1.plot(datas.coords[param_name], sig_thetas, 'g-')
ax1.errorbar(datas.coords[param_name], sig_thetas, color='g', fmt="o", label=r'sig_theta(deg)')
# ax1.plot(datas.coords[param_name], sig_phis, 'm-')
# ax1.errorbar(datas.coords[param_name], sig_phis, yerr=sig_phis_std, color='m', fmt="o", label=r'sig_phi(deg)')

ax0.set_xlabel(f'{param_name}')
ax0.set_ylabel('T1, T2 evolution time(ns)')
ax1.set_ylabel('Sig theta, phi(deg)')

ax0.legend(loc='best', ncols=2)
#plt.tight_layout(rect=[0, 0, 0, 0])
#fig.legend(ncols=2, fontsize='x-small', bbox_transform=fig.transFigure)
#plt.tight_layout()
plt.show()

"""--------------------Separate Information---------------------------------"""

# method ditches 'q4_ro'
dataset_data = xr.open_dataarray(fname)
print(dataset_data) #same as datas in forloop
"""
Reloaded modules: data_plots_qm, lm_utils, lineshapes_det_bench, lm_det_bench
Time taken: 7.538e+00 seconds
<xarray.DataArray 'q4_ro' (amp: 15, mixer: 1, sequence: 4, repeat_time: 800)>
[48000 values with dtype=float64]
Coordinates:
  * mixer        (mixer) object 'val'
  * sequence     (sequence) object 'T1' 'X X(T2 echo)' ... 'Y -Y (phase error)'
  * repeat_time  (repeat_time) int32 0 1 2 3 4 5 6 ... 794 795 796 797 798 799
  * amp          (amp) float64 0.06228 0.06328 0.06428 ... 0.07528 0.07628
"""

#print(dataset_data.coords["sequence"])
"""
<xarray.DataArray 'sequence' (sequence: 4)>
array(['T1', 'X X(T2 echo)', 'Y Y (rotational error)', 'Y -Y (phase error)'],
      dtype=object)
Coordinates:
  * sequence  (sequence) object 'T1' 'X X(T2 echo)' ... 'Y -Y (phase error)
"""
# print(dataset_data.coords["sequence"][0])
"""
<xarray.DataArray 'sequence' ()>
array('T1', dtype='<U2')
Coordinates:
    sequence  <U2 'T1'
"""
# print(dataset_data.coords["sequence"][1])
"""
<xarray.DataArray 'sequence' ()>
array('X X(T2 echo)', dtype='<U12')
Coordinates:
    sequence  <U12 'X X(T2 echo)'
"""
print('\n')
print(dataset_data.coords["sequence"].values)
"""
Retrieves experimental sequences
['T1' 'X X(T2 echo)' 'Y Y (rotational error)' 'Y -Y (phase error)']
"""
print('\n')

xdat = dataset_data.coords["repeat_time"].values
ydat = dataset_data.coords["amp"].values
print(ydat[0])
param_len = len(dataset_data.coords["amp"]) #
#print(param_len) #15
# only prints 1 value of the array

"""---------------------------------Dataset variables-----------------------"""

"""dataset.data_vars.items => variables"""
#print(list[dataset.data_vars.keys()])
"""
list[KeysView(Data variables:
    q4_ro    (amp, mixer, sequence, repeat_time) float64 0.792 0.811 ... 0.642)]
"""


# dataset_items = dataset.data_vars.items()
# # print(len(dataset_items)) # 1
# for ro_name, datas2 in dataset_items:
#     print(ro_name)
#     print(datas2)
#     """
#     datas2, core.dataarray.DataArray(15,1,4,800)
    
#     printed values all arrays
#     """
#     ydata2 = []
#     # need to know contents of nc format
#     param_len = len(datas2.coords[param_name])
    
#     for i in range(param_len):
#         ydata2.append([])
        
#         # unpack for 4 1D numpy arrays to one list from datas2
#         for data2 in datas2.values[i][0]:
#             ydata2[i].append(data2)
        
#         print(len(ydata2[i]))
#         for j in range(1, len(ydata2[i])):
#             # more compact way of analyzing for datasets
#             ydata2[i][j] = 1- ydata2[i][j]
#             # compact but not intuitive
        

        

# for ro_name, datas2 in dataset_itmes:
#     # 1 loop
#     param_len = len(datas2.coord['amp'])
#     for i in range(param_len):
#         ydata2=
print('\n')
print(len(range(1,4))) #3
#print(np.arange(0,15))