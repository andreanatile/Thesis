import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
import featrures_extraction as ft

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
location=pd.read_csv('data/smalldrive/Location.csv')
SaccY=Sacc['Acceleration y (m/s^2)']

segments=ft.Segmentation(SaccY,1000,0.66)

FFT_segments=[]
# Calculate the corresponding frequency values
sampling_rate = 400
frequencies = np.fft.fftfreq(1000, d=1/sampling_rate)

for segment in segments:
    FFT_segments.append(np.fft.fft(segment))

    
plt.figure(figsize=(10, 6))

# Plot the first Fourier Transform
plt.subplot(2, 1, 1)  # Subplot 1 (2 rows, 1 column, 1st position)
plt.plot(frequencies, np.abs(FFT_segments[50]))
plt.title('Fourier Transform of Acceleration y (m/s^2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  

# Plot the second Fourier Transform
plt.subplot(2, 1, 2)  # Subplot 2 (2 rows, 1 column, 2nd position)
plt.plot(frequencies, np.abs(FFT_segments[20]))
plt.title('Fourier Transform of Acceleration y (m/s^2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  

plt.tight_layout()

plt.show()