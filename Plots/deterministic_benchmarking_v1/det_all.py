from lm_det_bench import steps_db_errors

from xarray import open_dataset
dataset = open_dataset(r"D:\HW\Quela\QM\QM_data\5Q4C_20241016_2_AS1608\5Q4C_20241016_2_AS1608\20241218_165805_1QDB_all\1QDB_all.nc")
tg = 40 #ns 

for ro_name, data in dataset.data_vars.items():
    xdata = data.coords["repeat_time"].values
    ydata = []
    for data in data.values[0]:
        ydata.append(data)
        
    # Inverse the state into fidelity
    for i in range(1, len(ydata)):
        ydata[i] = 1- ydata[i]
        
    # Rescale
    for i in range(len(ydata)):
        arr_max = ydata[i].max()
        arr_min = ydata[i].min()
        ydata[i] = ydata[i]+1-arr_max # Max=1, only shift
        # ydata[i] = (ydata[i]-arr_min)/(arr_max - arr_min)
        

    fig2y_list, fig2_dict_dB = steps_db_errors(xdata_lst=[xdata*tg, xdata*tg,
                                                        xdata*tg, xdata*tg], 
                                                ydata_lst=[ydata[0], ydata[1],
                                                        ydata[2], ydata[3]], 
                                                tg=tg, show=['Y','Y'])
