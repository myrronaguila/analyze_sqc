# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:19:35 2025

Peak feature extraction in 1D plots and 2D maps

Motivation: 
    1. Feature extraction of massive datasets for the following applications:
        1.1. 2D map of Rabi Oscillations (detuning vs interaction time),
        1.2. Qubit / Coupler / Resonator Frequency Spectrum
        1.3. Strong-Coupling Physics Data (i.e. Rabi-splitting, EIT, OMIT, etc)
    2. Prescreening of datasets comprising of 1D or 2D maps prior to more 
        involved analysis like lmfit or scipy.optimize / scipy.curve_fit
        2.1. Note that curve-fitting algorithms are used for extract more than 1
        parameter that meets the eye, fine-tuned and robust data extraction
        with additional information of uncertainties.
        2.2. Avoid curve-fits on selected 1D plots in 2D map that would guarantee
        a failure in curve fitting due to bad SNR, interferences due to unknown
        physics and false positives.
    3. Allows template for programming-language agnostic application (if priority)
        is performance or statistics (i.e. ML).

20250329
    - Add statistical analysis of crosstalk matrix (not to relate with
                                                    confusion matrix)
    - from sklearn.decomposition import PCA

@author: Mai
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert, savgol_filter
from scipy.signal import find_peaks, peak_prominences
from sklearn.decomposition import PCA

"""---------------------1D-like peaks---------------------------------------"""
def classify_snr(time, voltage_signal, win_len=11, p_order=3):
    """Classify whether the SNR is high or low based on Hilbert Transform analysis.
    low_snr : hilbert transform / cwt
    medium_snr : hilbert transform 
    large_snr : fft
    
    """
    
    # Hilbert Transform
    analytic_signal = hilbert(voltage_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    
    # Compute Instantaneous Frequency
    dt = np.mean(np.diff(time))  # Time step
    inst_freq = np.gradient(phase, dt) / (2 * np.pi)  # Convert to Hz
    
    # Compute Signal and Noise Power
    signal_power = np.var(voltage_signal)
    smoothed_signal = savgol_filter(voltage_signal, window_length=win_len, 
                                    polyorder=p_order)
    noise_power = np.var(voltage_signal - smoothed_signal)
    
    # Compute SNR in dB
    snr = signal_power / noise_power if noise_power > 0 else np.inf
    snr_db = 10 * np.log10(snr)

    # Phase Stability Check (Higher variance in frequency → lower SNR)
    phase_stability = np.std(inst_freq) / np.mean(np.abs(inst_freq))
    
    # Classification Criteria
    if snr_db > 20 and phase_stability < 0.1:
        classification = "High SNR"
    elif snr_db > 10 and phase_stability < 0.2:
        classification = "Moderate SNR"
    else:
        classification = "Low SNR"

    # Print Results
    print(f"SNR (dB): {snr_db:.2f}")
    print(f"Phase Stability Metric: {phase_stability:.4f}")
    print(f"Classification: {classification}")

    # Plot Results
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    
    # Plot Original Signal
    ax[0].plot(time, voltage_signal, label="Voltage Signal", color="b")
    ax[0].plot(time, smoothed_signal, label="Smoothed Signal", color="r", linestyle="--")
    ax[0].set_ylabel("Voltage (a.u.)")
    ax[0].legend()
    ax[0].set_title(f"Signal and SNR Analysis ({classification})")

    # Plot Instantaneous Frequency
    ax[1].plot(time, inst_freq, label="Instantaneous Frequency", color="g")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_xlabel("Time (ns)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    return classification


"""-----------2D mapping feature extraction --------------------------------"""
def search_peaks_from_2D_map(Z, x_arr, y_arr, height=0.5):
    """
    Rapid search for peaks in 2D map using scipy.optimize
    
    Use-cases
    1. Search for lorentzian peaks prior to lmfit

    Parameters
    ----------
    Z : 2D numpy array
        2D map of x_array (V) vs y_array (Hz,MHz,GHz) (y_arr, x_arr) 
    x_arr : 1D numpy array
        x_array (i.e. Voltage)
    y_arr : 1D numpy array
        y-array (i.e. Frequency)
    height : float, optional
        height of peaks to adjust for peak-search. The default is 0.5.

    Returns
    -------
    voltage_peaks : 1D numpy array
        x-axis elements of coordinates of peaks.
    freq_peaks : 1D numpy array
        y-axis elements of coordinates of peaks.
    indices_x : 1D numpy array
        x-axis indices in Z map (idx,)
    indices_y : 1D numpy array
        y-axis indices in Z map
    """
    
    # Store detected peak frequencies
    detected_freqs = [] # useful as there are unknown peaks
    indices_x = []
    indices_y = []
    
    for v_idx, v in enumerate(x_arr):
        transmission_slice = Z[:, v_idx]  # Get 1D transmission at this voltage

        # Find peaks in transmission data (resonance dips in transmission)
        peaks, _ = find_peaks(transmission_slice, height)  # Adjust height threshold as needed

        if len(peaks) > 0:
            # Compute peak prominence to filter noise
            prominences = peak_prominences(transmission_slice, peaks)[0]
            best_peak_idx = peaks[np.argmax(prominences)]  # Select peak with highest prominence
            detected_freqs.append((v, y_arr[best_peak_idx])) # registers x and y_arr, can be separate objects
            indices_x.append(v_idx)
            indices_y.append(best_peak_idx)

    # Convert detected peaks into separate arrays
    x_peaks, y_peaks = zip(*detected_freqs)
    return x_peaks, y_peaks, indices_x, indices_y

"""---------------------hilbert-extraction----------------------------------"""
def calculate_2dft(input):
    """Transform Data from 2D array of voltage/time to fourier amplitude
    Good only for high SNR signals
    """
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def search_freq_rabi_fft(Z, x_arr, y_arr):
    """
    Quick search for frequencies

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    x_arr : 1D numpy array
        Detuning / Voltage.
    y_arr : 1D numpy array
        Interaction time.

    Returns
    -------
    None.

    """
    # Store detected peak frequencies
    detected_freqs = [] # useful as there are unknown peaks
    indices_x = []
    
    for i in range(len(x_arr)):
        data = Z[:,i]
        data = data - data.mean()
        frequencies = np.fft.fftfreq(len(y_arr), abs(y_arr[-1] - y_arr[0]) / (len(y_arr) - 1))
        fft = abs(np.fft.fft(data))
        argmax = abs(fft).argmax()
        amplitude = fft[argmax] / len(fft)
        frequency = abs(frequencies[argmax])
        detected_freqs.append(frequency)
    return np.array(detected_freqs)

def search_freq_rabi_hil(Z, x_arr, y_arr, thresh=0.5):
    """
    # Hilbert Transform to extract instantaneous frequency
    # it works but it has the noisy behavior like fft.rabi
    """
    insta_freq = []
    valid_x_arr = []
    idx_x_arr = []

    for i, freq in enumerate(x_arr):
        signal_data = Z[:,i]
    
        # Apply Hilbert transform
        analytic_signal = hilbert(signal_data)
        phase = np.unwrap(np.angle(analytic_signal)) 
        #phase-unwrap leads to complicaed behavior
    
        # Compute instantaneous frequency (derivative of phase)
        inst_freq = np.gradient(phase, y_arr)
    
        # Apply filtering: Keep only frequencies with clear oscillation trends
        median_freq = np.median(inst_freq)
        
        if np.abs(median_freq) > thresh:  # Threshold based on expected Rabi range
            insta_freq.append(median_freq)
            valid_x_arr.append(freq)
            idx_x_arr.append(i)
    
    return insta_freq, valid_x_arr, idx_x_arr

"""-----Statistics from Crosstalk Matrix / Confusion Matrix Features--------"""
def get_stats_from_crosstalk(Z, select_params=None):
    """
    Get Off-diagonal statistics from crosstalk matrix
    Assume a rectangular spectrum. Avoid using for-loops for stats.
    
    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # get dimensions of crosstalk matrix if not a rectangular dimension
    m = len(Z[:,0])
    n = len(Z[0,:])
    # list off-diagonal index from diag[0]
    int_diag = np.arange(1,m-1,1)
        
    # get diagonals as numpy arrays
    diag_elem = np.linalg.diagonal(Z, offset=0)
    diag_mat = np.diag(v=diag_elem, k=0)
    
    # Extract off-diagonal elements
    xtalk_offs = Z[~np.eye(Z.shape[0], dtype=bool)] #1D numpy array
    
    # get L1_norm, sparsity (total of crosstalk level made)
    L1_norm = np.sum(np.abs(xtalk_offs))
    # calculate average and std crosstalk
    xtalk_mean = L1_norm/(m*(n-1)) # assuming a square
    xtalk_std = np.std(xtalk_offs)
    
    # get minimum and maximum crosstalk
    xtalk_max = xtalk_offs.min()
    xtalk_min = xtalk_offs.max()
    
    # estimate sum of asymmetries
    off_diag_u = Z[np.triu_indices_from(Z, k=1)]
    off_diag_l = Z[np.tril_indices_from(Z, k=1)]
    assym_arr = np.array([np.abs(off_diag_u[i] - off_diag_l[i]) for i in range(len(off_diag_u))])
    assym_sum = np.sum(assym_arr)
    
    # get ECDF of crosstalk for probability distribution using lm.stats
        
    
    # 6. Principal Component Analysis (PCA) on Crosstalk Patterns
    # Step 2: Perform PCA on Crosstalk Matrix
    # subtract diagonal elements from array
    xtalk_mat = Z - diag_mat # matrix
    pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
    Z_pca = pca.fit_transform(xtalk_mat)
    
    # Step 3: Analyze the Results
    explained_variance = pca.explained_variance_ratio_  # Variance explained by each PC
    principal_components = pca.components_  # Principal component vectors

    # Print Results
    print("Explained Variance Ratio:", explained_variance)
    print("Principal Components:\n", principal_components)
        
    # get correlated device parameters with frequency or distance
    
    """9. Graph Representation and Community Detection
    One can represent the crosstalk matrix as a graph, where nodes are qubits and edges represent crosstalk strength.
    Graph clustering: Identifies groups of strongly coupled qubits, which could be targeted for localized crosstalk cancellation.
    Graph centrality measures: Identify qubits that are most affected by crosstalk."""
    
    dict_val={'xtalk_mean_std': [xtalk_mean,xtalk_std],
              'xtalk_min_max': [xtalk_min,xtalk_max],
              'xtalk_L1_norm': [L1_norm, 0],
              'xtalk_assym': [assym_sum,0],
              'xtalk_pca_var': explained_variance,
              'xtalk_pca_exp': principal_components}
    return dict_val

def get_fidelity_from_matrix(Z, select_params):
    
    return