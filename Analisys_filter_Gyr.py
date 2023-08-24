import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

# Read the CSV Gyroscope file 
data_Gyr = pd.read_csv('data/fermo/Gyroscope.csv')
data_Gyr=data_Gyr[(data_Gyr['Time (s)']<5)&(data_Gyr['Time (s)']>2)]

print(data_Gyr.head())

#--------------- Calculate the Fourier Transform of z-axis Gyroscope--------------------
fft_result_Gyr_z = np.fft.fft(data_Gyr["Gyroscope z (rad/s)"])

# Calculate the corresponding frequency values
sampling_rate = 400
frequencies = np.fft.fftfreq(len(data_Gyr["Gyroscope z (rad/s)"]), d=1/sampling_rate)
index=np.argmax(np.abs(fft_result_Gyr_z))
max_amplitude_frequency=frequencies[index]
# Plot the original signal and its Fourier Transform
plt.figure(figsize=(10, 6))

# Plot the original signal (optional)
plt.subplot(2, 1, 1)
plt.plot(data_Gyr['Time (s)'],data_Gyr["Gyroscope z (rad/s)"])
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_result_Gyr_z))
plt.title('Fourier Transform of z-axis Gyroscope')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  # Display only positive frequencies
plt.tight_layout()

plt.show()
print(max_amplitude_frequency)

#---------------------calculate the Fourier Trasform of y-axis Gyroscopo-----------

fft_result_Gyr_y = np.fft.fft(data_Gyr["Gyroscope y (rad/s)"])

# Plot the original signal and its Fourier Transform
plt.figure(figsize=(10, 6))

# Plot the original signal (optional)
plt.subplot(2, 1, 1)
plt.plot(data_Gyr['Time (s)'],data_Gyr["Gyroscope y (rad/s)"])
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fft_result_Gyr_y))
plt.title('Fourier Transform of y-axis Gyroscope')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  # Display only positive frequencies
plt.tight_layout()

plt.show()