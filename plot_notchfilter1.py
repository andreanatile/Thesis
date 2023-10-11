import numpy as np
import pandas as pd
from notch_filter import notch_filter_data
from featrures_extraction import Segmentation
import matplotlib.pyplot as plt

segment_length = 1000
sampling_rate = 400
overlap_percentage = 0.66
strada_liscia = pd.read_csv("data\strada liscia\Accelerometer.csv")
Yacc = strada_liscia['Acceleration y (m/s^2)']

filtered_segments = notch_filter_data(Yacc, segment_length, sampling_rate, overlap_percentage)

raw_segments = Segmentation(Yacc, segment_length, overlap_percentage)

fft_raw_segments = []
fft_filtered_segments = []

for i in range(0, len(filtered_segments)):
    # Eliminating offset
    raw_segments[i]=raw_segments[i]- np.mean(raw_segments[i])
    filtered_segments[i]=filtered_segments[i]- np.mean(filtered_segments[i])
    # FFT
    fft_raw_segments.append(np.fft.fft(raw_segments[i]))
    fft_filtered_segments.append(np.fft.fft(filtered_segments[i]))

# Create a list of titles for each signal
titles = [
    'FFT of raw segment 11',
    'FFT of raw segment 12',
    'FFT of raw segment 13',
    'FFT of filtered segment 11',
    'FFT of filtered segment 12',
    'FFT of filtered segment 13'
]

# Create a 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Signals to plot
signals = [fft_raw_segments[11], fft_raw_segments[12], fft_raw_segments[13], fft_filtered_segments[11],
           fft_filtered_segments[12], fft_filtered_segments[13]]
freq = np.fft.fftfreq(segment_length, d=1/sampling_rate)

# Generate and plot FFT for each signal
for i, signal in enumerate(signals):
    

    # Plot FFT magnitude
    ax = axes[i // 3, i % 3]  # Adjusted indexing here
    ax.plot(freq, 2*np.abs(signals[i])/len(signals[i]))
    ax.set_title(titles[i],fontsize=16)  # Use the defined titles
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude (m/s^2)')
    ax.set_xlim(0, sampling_rate / 2)
    ax.set_ylim(0,0.44)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # Add vertical lines for every 20 Hz interval
    for frequency in range(0, int(sampling_rate / 2) + 1, 20):
        ax.axvline(x=frequency, color='red', linestyle='--')

# Adjust layout and display
plt.tight_layout()
plt.savefig("plot/11 12 13 new.png")
plt.show()
