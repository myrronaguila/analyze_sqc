# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:46:39 2021
Library for Visualization of Data plots
@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec

"""Adjusting Norm in Twin X data"""
# import matplotlib.colors as mcolors
# from matplotlib.colors import TwoSlopeNorm

# from mpl_toolkits.axes_grid1 import make_axes_locatable

# from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.colors import Normalize

"""-------------------Organize Plot Directions------------------------------"""



"""-------------------making plots from axes----------------------------
Save, show legends and show will be done after each axes.
"""

"""-----------------------Format Plot Styles--------------------------------"""
# standard color-pallet (focused on distinguishability)
col_stand = ['tab:blue', 'tab:orange', 'tab:green', 'tab:read', 'tab:purple',
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:cyan', 'k']

def assign_rand_color(num_col):
    """
    Assign random color based on assigned numbers

    Parameters
    ----------
    num_col : float
        number of needed colors.

    Returns
    -------
    color_arr . list of strings
        colors in hexadecimal structure
    """
    # get number of random colors from hexadecimal pairs
    hexadecimal_alphabets = '0123456789ABCDEF'
    color_arr = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in 
                                range(6)]) for i in range(num_col)]
    return color_arr

col_stand = assign_rand_color(10)
# standard marker style
col_mk = ['.' for i in range(len(col_stand))]
col_ls = ['solid' for i in range(len(col_stand))]
col_cap = [5 for i in range(len(col_stand))]
#std_fmt_cmlcap = [[col]]

def cm_to_inch(x):
    return x/2.54

def set_font_default(font_size):
    """
    Set font size to figures

    Parameters
    ----------
    font_size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rc('xtick', labelsize=font_size) 
    plt.rc('ytick', labelsize=font_size)
    plt.rcParams.update({'font.size': font_size})
    return

"""-------------------------Basic Plot Tools--------------------------------"""

def Quick_Plot(x1, y1, x2, y2, curve_label, axes_label, title):
    plt.title(title)
    plt.plot(x1, y1, 'ko', label=curve_label[0]) 
    plt.plot(x2, y2, 'r-.', label=curve_label[1])
    plt.xlabel(axes_label[0])
    plt.ylabel(axes_label[1])
    plt.legend(loc='best')
    """We can put an optional save-tool here for data analysis"""
    plt.show()
    return

def line_plot(ax, x_dat, y_dat, axs_lbl, lbl_arr, col_mark, ticks_2D,
              font_size):
    """
    Line plot only. Disadvantage of this plot is that this is too dependent
    on col_mark. Used in Aguila's paper and thesis but turned out to be not
    scalable due to col_mark
    
    comments: 20240330 - col_mark seemed too complicated. 
    Key rules: if col_mark is not defined, then use default marks
            

    Parameters
    ----------
    ax : axes
        ax.
    x_dat : list of 1D numpy array
        list of x_data.
    y_dat : list of 1D numpy array
        list of y_data.
    axs_lbl : list of strings
        axs_lbl[0] = x_label.
        axs_lbl[1] = y_label.
    lbl_arr : list of strings
        list of each label per x_dat
        (must be the same length as elements in x_dat).
    col_mark : list of list of marker labels
        each element in list = [color, marker, linestyle, capsize]
    ticks_2D: list of 1D numpy array
        ticks_2D[0] = xticks
        ticks_2D[1] = yticks
    font_size : integer
        size of phont

    Returns
    -------
    axes.
    """
    
    #set font-size as default
    set_font_default(font_size=font_size)
    
    """Plot array"""
    n = len(lbl_arr)
    # print(n)
    
    if col_mark == []:
        # use standard python pallets
        [ax.plot(x_dat[i], y_dat[i], label=lbl_arr[i]) for i in range(n)]
    else:
        [ax.plot(x_dat[i], y_dat[i], color=col_mark[i][0],  marker=col_mark[i][1], 
                 linestyle=col_mark[i][2], label=lbl_arr[i]) for i in range(n)]
    
    """Set Labels"""
    ax.set_xlabel(axs_lbl[0], labelpad=-1)
    ax.set_ylabel(axs_lbl[1], labelpad=-1)
    
    """Set ticks based on input criteria"""
    if len(ticks_2D[0]) > 0:
        ax.set_xticks(ticks_2D[0])
    if len(ticks_2D[1]) > 0:
        ax.set_yticks(ticks_2D[1])
        
    ax.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                   right = True)
    
    return

def error_plot(ax, x_dat, y_dat, x_err, y_err, axs_lbl, lbl_arr, col_mark, 
               ticks_2D, font_size):
    """
    Error Bar Plot Only

    Parameters
    ----------
    ax : axes
        ax.
    x_dat : list of 1D numpy array
        list of x_data.
    y_dat : list of 1D numpy array
        list of y_data.
    x_err : list of 1D numpy array
        list of x_error.
    y_err : list of 1D numpy array
        list of y_error.
    axs_lbl : list of strings
        axs_lbl[0] = x_label.
        axs_lbl[1] = y_label.
    lbl_arr : list of strings
        list of each label per x_dat
        (must be the same length as elements in x_dat).
    col_mark : list of list of marker labels
        each element in list = [color, marker, linestyle, capsize]
    ticks_2D: list of 1D numpy array
        ticks_2D[0] = xticks
        ticks_2D[1] = yticks
    font_size : integer
        size of phont

    Returns
    -------
    axes.
    """
    
    set_font_default(font_size=font_size)
    # has an error
    # TypeError: plot got an unexpected keyword argument 'x'
    """Plot array"""
    n = len(lbl_arr)
    if col_mark == []:
        [ax.errorbar(x=x_dat[i], y=y_dat[i], xerr=x_err[i], yerr=y_err[i],
                     capsize=5, marker='o', linestyle=None, 
                     label=lbl_arr[i]) for i in range(n)]
    else: 
        [ax.errorbar(x=x_dat[i], y=y_dat[i], xerr=x_err[i], yerr=y_err[i],
                     capsize=col_mark[i][3], color=col_mark[i][0], 
                     marker=col_mark[i][1], linestyle=col_mark[i][2], 
                     label=lbl_arr[i]) for i in range(n)]
    
    """Set Labels"""
    ax.set_xlabel(axs_lbl[0], labelpad=-1)
    ax.set_ylabel(axs_lbl[1], labelpad=-1)
    
    """Set ticks based on input criteria"""
    if len(ticks_2D[0]) > 0:
        ax.set_xticks(ticks_2D[0])
    if len(ticks_2D[1]) > 0:
        ax.set_yticks(ticks_2D[1])
    
    """Tick Inset"""
    ax.tick_params(axis = 'both', which ='both', direction='in', top=True, 
                   right = True)
    return

def line_plot_3D(ax, x_dat, y_dat, z_dat, axs_lbl, lbl_arr, col_mark, ticks_3D,
                 pad_dat, view_dat, font_size):
    """
    3D line plot

    Parameters
    ----------
    ax : axes
        ax.
    x_dat : list of 1D numpy array
        list of x_data.
    y_dat : list of 1D numpy array
        list of y_data.
    axs_lbl : list of strings
        axs_lbl[0] = x_label.
        axs_lbl[1] = y_label.
    lbl_arr : list of strings
        list of each label per x_dat
        (must be the same length as elements in x_dat).
    col_mark : list of list of marker labels
        each element in list = [color, marker, linestyle]
    ticks_3D: list of 1D numpy array
        ticks_3D[0] = xticks
        ticks_3D[1] = yticks
        ticks_3D[2] = zticks
    pad_dat : list of integers
        pad_dat[0] = x-label pads
        pad_dat[1] = y-label pads 
        pad_dat[2] = z-label pads 
        pad_dat[3] = xticks pad
        pad_dat[4] = yticks pad
        pad_dat[5] = zticks pad
    view_dat : list of integers
        view_dat[0] = azim
        view_dat[1] = elev
        view_dat[2] = zoom
        
    Returns
    -------
    axes.
    """
    #At 3D axes, no need to label line colors
    set_font_default(font_size=font_size)
    
    [ax.plot(xs=x_dat[i][:], ys=y_dat[i][:], zs=z_dat[i], 
          zdir='x', color=col_stand[i], marker=col_mark[i][1],
          linestyle=col_mark[i][2]) for i in range(len(z_dat))]
    
    """Put labels"""
    ax.set_xlabel(axs_lbl[0], labelpad=pad_dat[0])
    ax.set_ylabel(axs_lbl[1], labelpad=pad_dat[1])
    ax.set_zlabel(axs_lbl[2], labelpad=pad_dat[2])
    
    """Put x, y and z ticks"""
    if len(ticks_3D[0]) > 0:
        ax.set_xticks(ticks_3D[0])
    if len(ticks_3D[1]) > 0:
        ax.set_yticks(ticks_3D[1])
    if len(ticks_3D[1]) > 0:
        ax.set_zticks(ticks_3D[2])
    
    """tick_params Pad Data"""
    ax.tick_params('x', pad=pad_dat[3])
    ax.tick_params('y', pad=pad_dat[4])
    ax.tick_params('y', pad=pad_dat[5])
    
    """Adjust View"""
    ax.view_init(azim=view_dat[0], elev=view_dat[1])
    ax.dist(view_dat[2])
    return

def color_map(fig, ax, x_dat, y_dat, z_dat, axs_lbl, cmap, ticks_2D, pad_dat, 
              font_size, prop_colbar, norm='N'):
    """
    2D color map

    Parameters
    ----------
    fig : figure
        figure of drawing
    ax : axes
        ax.
    x_dat : 1D numpy array
        x_data.
    y_dat : 1D numpy array
        y_data.
    z_dat : 2D numpy array
        z_data.
    axs_lbl : list of strings
        axs_lbl[0] = x_label.
        axs_lbl[1] = y_label.
    cmap : string
        color map.
    ticks_2D : list of 1D numpy array
        ticks_2D[0] = xticks
        ticks_2D[1] = yticks
        ticks_2D[2] = zticks
    pad_dat : TYPE
        pad_dat[0] = x-label pads
        pad_dat[1] = y-label pads 
        pad_dat[2] = xticks pad
        pad_dat[3] = yticks pad
        pad_dat[4] = z-label pads
    font_size : integer
        font size.
    prop_colbar : list of list and string
        color bar within 2D map
        prop_colbar[0] = [x_coord, y_coord, width fraction, length fraction]
        prop_colbar[1] = 'horizontal' or vertical
    norm : string
        norm='Y' => set to zero value

    Returns
    -------
    None.

    """
    
    set_font_default(font_size=font_size)
    
    """Plot Data"""
    if norm=='Y':
        """Setting data"""
        """TwoSlopeNorm non-existent in Matplotlib 3.1.3"""
        # norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=np.amin(z_dat), vmax=np.amax(z_dat))
        # im = ax.pcolormesh(x_dat, y_dat, z_dat, cmap=cmap, norm=norm, 
        #                    vmin=np.amin(z_dat), vmax=np.amax)
        """Matplotlib 3.1.3."""
        if np.abs(np.amin(z_dat)) > np.abs(np.amax(z_dat)):
            max_val = np.abs(np.amin(z_dat))
        else:
            max_val = np.abs(np.amax(z_dat))
        im = ax.pcolormesh(x_dat, y_dat, z_dat, cmap=cmap, 
                           vmin=-1*max_val, vmax=max_val) 
    else:
        im = ax.pcolormesh(x_dat, y_dat, z_dat, cmap=cmap)
    
    """Values centered at zero value"""
    
    #ax.set_aspect(aspect=1)
    
    """Analyze"""
    ax.set_xlabel(axs_lbl[0], labelpad=pad_dat[0])
    ax.set_ylabel(axs_lbl[1], labelpad=pad_dat[1])
    
    """Set ticks based on input criteria"""
    if len(ticks_2D[0]) > 0:
        ax.set_xticks(ticks_2D[0])
    if len(ticks_2D[1]) > 0:
        ax.set_yticks(ticks_2D[1])
    
    """Set scale bar outside ax1"""
    """ 
    https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
    """
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    
    """Put color bar with ax1 but smaller"""
    # divider = make_axes_locatable(ax)
    cax = fig.add_axes(prop_colbar[0])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    if len(pad_dat) < 5:
        fig.colorbar(im, cax=cax, orientation=prop_colbar[1], label=axs_lbl[2])
    else:    
        fig.colorbar(im, cax=cax, orientation=prop_colbar[1], label=axs_lbl[2],
                     pad=pad_dat[4], ticks=ticks_2D[2])
    
    """ticks_params Pad Data"""
    ax.tick_params('x', pad=pad_dat[2])
    ax.tick_params('y', pad=pad_dat[3])
    ax.tick_params(axis='both', which='major', direction='in', top=True, 
                   right=True, labelsize=font_size)
    return

def plot_3D(ax, x_dat, y_dat, z_dat, axs_lbl, lbl_arr, cmap, ticks_3D, 
            pad_dat, view_dat, font_size):
    """
    spatial 3D color plot

    Parameters
    ----------
    ax : axes
        ax.
    x_dat : 1D numpy array
        list of x_data.
    y_dat : 1D numpy array
        list of y_data.
    z_dat : 2D numpy array
        list of z_data in xy plane.
    axs_lbl : list of strings
        axs_lbl[0] = x_label.
        axs_lbl[1] = y_label.
    lbl_arr : list of strings
        list of each label per x_dat
        (must be the same length as elements in x_dat).
    cmap : string
        cmap
    ticks_3D: list of 1D numpy array
        ticks_3D[0] = xticks
        ticks_3D[1] = yticks
        ticks_3D[2] = zticks
    pad_dat : list of integers
        pad_dat[0] = x-label pads
        pad_dat[1] = y-label pads 
        pad_dat[2] = z-label pads 
        pad_dat[3] = xticks pad
        pad_dat[4] = yticks pad
        pad_dat[5] = zticks pad
    view_dat : list of integers
        view_dat[0] = azim
        view_dat[1] = elev
        view_dat[2] = zoom
        
    Returns
    -------
    axes.
    """
    set_font_default(font_size=font_size)
    
    """Create 2-D spatial grid"""
    X, Y = np.meshgrid(x_dat, y_dat)
    Z = z_dat.to_numpy()
    
    """Plot Data"""
    ax.plot_surface(Y, X, Z, rstride=1, cstride=1, cmap=cmap, edgecolor='none')
    
    """Put labels"""
    ax.set_xlabel(axs_lbl[0], labelpad=pad_dat[0])
    ax.set_ylabel(axs_lbl[1], labelpad=pad_dat[1])
    ax.set_zlabel(axs_lbl[2], labelpad=pad_dat[2])
    
    """tick_params Pad Data"""
    ax.tick_params('x', pad=pad_dat[3])
    ax.tick_params('y', pad=pad_dat[4])
    ax.tick_params('y', pad=pad_dat[5])
    
    """Put x, y and z ticks"""
    if len(ticks_3D[0]) > 0:
        ax.set_xticks(ticks_3D[0])
    if len(ticks_3D[1]) > 0:
        ax.set_yticks(ticks_3D[1])
    if len(ticks_3D[1]) > 0:
        ax.set_zticks(ticks_3D[2])
    
    """Adjust View"""
    ax.view_init(azim=view_dat[0], elev=view_dat[1])
    ax.dist(view_dat[2])
    return

def vertbar_plot(ax, x_dat, y_dat, axs_lbl, label_arr, font_size):
    """
    Useful for T1 and T2 distribution for errors

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    x_dat : TYPE
        DESCRIPTION.
    y_dat : TYPE
        DESCRIPTION.
    axs_lbl : TYPE
        DESCRIPTION.
    label_arr : TYPE
        DESCRIPTION.
    font_size : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    return

def bar_3d_2q(ax, dz_exp, dz_mod, w_bar, label_arr, font_size, fig_pars):
    """
    Bar graph data for two-qubit interaction
    Useful for 2-qubit state Bell-state and 2qubit operation state
    
    Designing Bar graphs
    References: https://pythonprogramming.net/3d-bar-chart-matplotlib-tutorial/
    references: https://stackoverflow.com/questions/24736758/parameters-required-by-bar3d-with-python
    
    Add color to bar graphs:
        References: https://stackoverflow.com/questions/50203580/how-to-use-matplotlib-to-draw-3d-barplot-with-specific-color-according-to-the-ba
    
    Adjust colorbar size and label:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        https://stackoverflow.com/questions/18403226/matplotlib-colorbar-background-and-label-placement
        
        text rotation: https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text
    
    Changing z-axis spine
        https://stackoverflow.com/questions/15042129/changing-position-of-vertical-z-axis-of-3d-plot-matplotlib
    
    Overlapping Bar Chart
    
    Parameters
    ----------
    ax : axes
        Projection must be defined as 3D.
    dz_exp : 2D numpy array
        4 x 4 matrix of measured states
    dz_mod : 2D numpy array
        4 x 4 matrix of axis states        
    label_arr : List of strings
        label_arr[0] = label of states/operation.
        label_arr[1] = label of color bar.
    font_size : float
        Font size.
    fig_pars : list of items for color bar
        fig_pars[0] = tick list
        fig_pars[1] = shrink of color bar
        fig_pars[2] = aspect ratio
        fig_pars[3] = pad
        fig_pars[4] = azim
        fig_pars[5] = view_dist
        
    Returns
    -------
    None.

    """
    
    """Set font size"""
    set_font_default(font_size)

    """Set-up documentation"""    
    x = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]) # x coordinates of each bars
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]) # y coordinates of each bars 
    z = np.zeros(16) #z coordinates of each bars

    """Adjust area"""
    x_adj = np.asarray([x[i] - (w_bar/2) for i in range(len(x))]) #readjust bar location according to w_bar
    y_adj = np.asarray([y[i] - (w_bar/2) for i in range(len(x))]) #readjust bar location according to w_bar

    dx = np.ones(len(x))*w_bar
    dy = np.ones(len(y))*w_bar
    # dz = np.ones(z) 
    
    """Adjust color of parameters"""
    cmap = cm.get_cmap('jet')
    norm = Normalize(vmin=min(dz_exp), vmax=max(dz_exp))
    colors = cmap(norm(dz_exp))
    
    """Plot axes bar with the theoretical data"""
    ax.bar3d(x_adj, y_adj, z, dx, dy, dz_mod, color='w', zsort='average', 
             alpha=0.1, ec='k', ls='dashed')
        
    """Plot axes bar with the experimental data"""
    ax.bar3d(x_adj, y_adj, z, dx, dy, dz_exp, color=colors, zsort='average', 
             alpha=0.5, ec='k', ls='solid')
    
    """Set tick limits to z-axis"""
    ax.set_zlim(fig_pars[0][0],fig_pars[0][-1])
    
    """Replace ticks to states / Operations"""
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_zticks(fig_pars[0])

    ax.set_xticklabels(label_arr[0])
    ax.set_yticklabels(label_arr[0])
    
    """Show colorbar as separate entity as part of the bar3d plot"""
    sc = cm.ScalarMappable(cmap=cmap,norm=norm)
    sc.set_array([])
    sc.set_clim(vmin=fig_pars[0][0], vmax=fig_pars[0][-1])
    # cb = plt.colorbar(sc, label=label_arr[1], ticks=[0, 0.5, 1], 
    #                   shrink=0.5, aspect=10, pad=-0.02)
    cb = plt.colorbar(sc, ax = ax, label=label_arr[1], ticks=fig_pars[0], 
                        shrink=fig_pars[1], aspect=fig_pars[2], pad=fig_pars[3])
    cb.set_label(label_arr[1], labelpad=-25, y=1.2, rotation='horizontal')

    """Fix z-axis plane """
    tmp_planes = ax.zaxis._PLANES 
    ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                        tmp_planes[0], tmp_planes[1], 
                        tmp_planes[4], tmp_planes[5])
    view_2 = (fig_pars[4], -45)
    init_view = view_2
    ax.view_init(*init_view)
    # ax.dist(fig_pars[5])
    return ax

"""--------------------------Plot Hierarchies-------------------------------"""


"""-------------------------Plot Animation----------------------------------"""


"""-------------------------------Test Place--------------------------------"""
