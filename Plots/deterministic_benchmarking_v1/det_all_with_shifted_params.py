from lm_det_bench import steps_db_errors

from xarray import open_dataset
dataset = open_dataset(r"D:\HW\Quela\QM\QM_data\5Q4C_20241016_2_AS1608\5Q4C_20241016_2_AS1608\Using_result_20241206\q4\20241219_171357_SQDB_shift_one_param\SQDB_shift_one_param.nc")
tg = 40 #ns 
param_name = 'amp'

# print(dataset)
for ro_name, datas in dataset.data_vars.items():
    xdata = datas.coords["repeat_time"].values
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
            
        # Inverse the state into fidelity
        for j in range(1, len(ydata[i])):
            ydata[i][j] = 1- ydata[i][j]
            
        # Rescale
        for j in range(len(ydata[i])):
            arr_max = ydata[i][j].max()
            arr_min = ydata[i][j].min()
            # ydata[i][j] = (ydata[i][j]-arr_min)/(arr_max - arr_min) # Max=1, min=0
            ydata[i][j] = ydata[i][j]+1-arr_max # Max=1, only shift
            

        y_list, dict_dB = steps_db_errors(xdata_lst=[xdata*tg*2, xdata*tg*2,
                                                            xdata*tg*2, xdata*tg*2], 
                                                    ydata_lst=[ydata[i][0], ydata[i][1],
                                                            ydata[i][2], ydata[i][3]], 
                                                    tg=tg, show=['N','N'])
        y_lists.append(y_list)
        T1s.append(dict_dB['T1_ns'][0])
        T1s_std.append(dict_dB['T1_ns'][1])
        T2s.append(dict_dB['T2r_ns'][0])
        T2s_std.append(dict_dB['T2r_ns'][1])
        sig_thetas.append(dict_dB['sig_theta_deg'][0])
        sig_thetas_std.append(dict_dB['sig_theta_deg'][1])
        sig_phis.append(dict_dB['sig_phi_deg'][0])
        sig_phis_std.append(dict_dB['sig_phi_deg'][1])
        
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from data_plots_qm import cm_to_inch       
wfig=8.6
fig = plt.figure(constrained_layout=True, figsize=(2*cm_to_inch(wfig),
                                                    2*cm_to_inch(wfig)))
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, hspace =0.1, wspace=0.1)
ax0 = fig.add_subplot(spec[0, 0]) # line-plot from literature and compare with model
# ax0.plot(datas.coords[param_name], T1s, 'k-')
# ax0.errorbar(datas.coords[param_name], T1s, yerr=T1s_std, color='k', fmt="s", label=r'T1(ns)')
ax0.plot(datas.coords[param_name], T2s, 'b-')
ax0.errorbar(datas.coords[param_name], T2s, yerr=T2s_std, color='b', fmt="s", label=r'T2(ns)')
ax1 = ax0.twinx()
# ax1.plot(datas.coords[param_name], sig_thetas, 'g-')
# ax1.errorbar(datas.coords[param_name], sig_thetas, color='g', fmt="o", label=r'sig_theta(deg)')
# ax1.plot(datas.coords[param_name], sig_phis, 'm-')
# ax1.errorbar(datas.coords[param_name], sig_phis, yerr=sig_phis_std, color='m', fmt="o", label=r'sig_phi(deg)')

ax0.set_xlabel(f'{param_name}')
ax0.set_ylabel('T1, T2 evolution time(ns)')
ax1.set_ylabel('Sig theta, phi(deg)')

fig.legend(ncols=2, fontsize='x-small', bbox_transform=fig.transFigure)
plt.tight_layout(rect=[0, 0, 0, 0])
plt.show()
