import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

Sacc=pd.read_csv("data/strada liscia/Accelerometer.csv")
Sgyr=pd.read_csv("data/strada liscia/Gyroscope.csv")

#-------------FFT for the vertical acceleration y-------------------------------------
pd.set_option('display.max_columns', None)
#print(data.head())


# Calculate the Fourier Transform of accelerometer y axis
fft_result = np.fft.fft(Sacc["Acceleration y (m/s^2)"])

# Calculate the corresponding frequency values
sampling_rate = 100
frequencies = np.fft.fftfreq(len(Sacc["Acceleration y (m/s^2)"]), d=1/sampling_rate)
index=np.argmax(np.abs(fft_result))
max_amplitude_frequency=frequencies[index]
# Plot the original signal and its Fourier Transform
plt.figure(figsize=(10, 6))

# Plot the original signal (optional)
plt.subplot(2, 1, 1)
plt.plot(Sacc['Time (s)'],Sacc["Acceleration y (m/s^2)"])
plt.title('Original Signal of Acceleration y (m/s^2)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_result))
plt.title('Fourier Transform of Acceleration y (m/s^2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.ylim(0,200)
plt.xlim(0,200)  
plt.tight_layout()

plt.show()