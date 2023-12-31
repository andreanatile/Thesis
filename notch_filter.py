import numpy as np
from scipy import signal
from featrures_extraction import Segmentation


def notch_filter_segment(current_segmentFFT, previous_max_freq_index, pervious_max_amplitude, sampling_rate, overlap_percentage):

    # Define parameters for the analysis
    band_size = 20  # Frequency band size in Hz
    frequency_resolution = sampling_rate/len(current_segmentFFT)  

    window_size = int(band_size*len(current_segmentFFT) /
                      sampling_rate)  
    # Step size for overlapping windows
    step = int(window_size * (1 - overlap_percentage))  

    # Initialize lists to store energy in each band
    energy_in_bands = []

    # Calculate energy for each frequency band using overlapping windows
    for i in range(1, int(len(current_segmentFFT)/2) - window_size + 1, step):
        window_data = current_segmentFFT[i:i+window_size]

        # Calculate the energy in each band
        energy_band = np.sum(np.abs(window_data)**2)
        energy_in_bands.append(energy_band)

    max_freq_index = np.argmax(energy_in_bands)
    max_energy = np.max(energy_in_bands)

    if (abs(max_freq_index-previous_max_freq_index) <= 2) & (abs(max_energy-pervious_max_amplitude) <= 0.4*pervious_max_amplitude):
        f1 = step*max_freq_index*frequency_resolution
        f2 = f1+band_size

        # Frequency to remove from the signal,0 < w0 < 1, with w0 = 1 corresponding to half of the sampling frequency.
        f0 = (f1+f2)/2
        w0 = f0

        # characterizes notch filter -3 dB bandwidth bw relative to its center frequency, Q = w0/bw
        Q = w0/band_size

        # Return the numerator(b) and denominator(a) polynomials of the IIR filter
        b, a = signal.iirnotch(w0, Q, sampling_rate)

        return max_freq_index, max_energy, b, a
    else:
        return max_freq_index, max_energy, None, None

# --------------------------------------------------------------------------

def notch_filter_data(data, segment_length, sampling_rate, overlap_percentage):
    # Segmenting the data
    segments = Segmentation(data, segment_length, overlap_percentage)

    # Calculating the FFT trasform for every segment
    FFT_Segments = []
    filtered_Segments = []
    for segment in segments:
        FFT_Segments.append(np.fft.fft(segment))

    for i in range(0, len(FFT_Segments)):
        # In case it's the first segment, set the previous frequency 
        # of the max band and amplitude at 0
        if i == 0:
            previous_max_freq_index, prev_max_energy, b, a = notch_filter_segment(
                FFT_Segments[0], 0, 0, sampling_rate, 0.66)
        else:
            max_freq_index, max_energy, b, a = notch_filter_segment(
                FFT_Segments[i], previous_max_freq_index, prev_max_energy, sampling_rate, 0.66)
            previous_max_freq_index, prev_max_energy = max_freq_index, max_energy

        if b is not None:
            filtered_segment = signal.filtfilt(b, a, segments[i])
            filtered_Segments.append(filtered_segment)
            print("segment filtered:" + str(i))

        else:
            filtered_Segments.append(segments[i])

    # Return the filtered segments
    return filtered_Segments
