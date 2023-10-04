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
    'FFT of raw segment 17',
    'FFT of raw segment 18',
    'FFT of filtered segment 17',
    'FFT of filtered segment 18'
]

# Create a 2x3 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Signals to plot
signals = [fft_raw_segments[17], fft_raw_segments[18],
           fft_filtered_segments[17], fft_filtered_segments[18]]
freq = np.fft.fftfreq(segment_length, d=1/sampling_rate)

# Generate and plot FFT for each signal
for i, signal in enumerate(signals):
    

    # Plot FFT magnitude
    ax = axes[i // 2, i % 2]  # Adjusted indexing here
    ax.plot(freq, np.abs(signals[i]))
    ax.set_title(titles[i])  # Use the defined titles
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(0, sampling_rate / 2)
    ax.set_ylim(0,160)

# Adjust layout and display
plt.tight_layout()
plt.show()
