import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pywt



def notch_filter(current_segmentFFT,previous_max_freq_index,pervious_max_amplitude,sampling_rate,overlap_percentage):
    
    # Define parameters for the analysis
    band_size = 10  # Frequency band size in Hz

    window_size = int(band_size*len(current_segmentFFT)/sampling_rate)  # Size of each analysis window
    step = int(window_size * (1 - overlap_percentage))  # Step size for overlapping windows

    # Initialize lists to store energy in each band
    energy_in_bands = []

    # Calculate energy for each frequency band using overlapping windows
    for i in range(1, int(len(current_segmentFFT)/2)- window_size + 1, step):
        window_data = current_segmentFFT[i:i+window_size]
            
        # Calculate the energy in each band
        energy_band = np.sum(np.abs(window_data)**2)
        energy_in_bands.append(energy_band)
        flag=i
    """ print('i:'+str(flag))
    print("widow size:"+ str(window_size)) """
    
    max_freq_index=np.argmax(energy_in_bands)
    max_energy=np.max(energy_in_bands)

    if (abs(max_freq_index-previous_max_freq_index)<=1) & (abs(max_energy-pervious_max_amplitude)<=0.2*pervious_max_amplitude):
        f1=step*max_freq_index
        f2=f1+band_size
        print(max_freq_index)
        w0=(f1+f2)/sampling_rate #Frequency to remove from the signal,0 < w0 < 1, with w0 = 1 corresponding to half of the sampling frequency.
        #characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw
        Q=w0/band_size

        b,a=signal.iirnotch(w0,Q,sampling_rate)
        return max_freq_index,max_energy,b,a
    else:
        return max_freq_index,max_energy,None,None
        



