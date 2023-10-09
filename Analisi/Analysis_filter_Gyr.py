import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

# Read the CSV Gyroscope file 
data_Gyr = pd.read_csv('data/fermo/Gyroscope.csv')
data_Gyr=data_Gyr[(data_Gyr['Time (s)']<5)&(data_Gyr['Time (s)']>2)]

#--------------- Calculate the Fourier Transform of z-axis Gyroscope--------------------
fft_result_Gyr_z = np.fft.fft(data_Gyr["Gyroscope z (rad/s)"])

# Calculate the corresponding frequency values
sampling_rate = 400
frequencies = np.fft.fftfreq(len(data_Gyr["Gyroscope z (rad/s)"]), d=1/sampling_rate)
index=np.argmax(np.abs(fft_result_Gyr_z))
max_amplitude_frequency=frequencies[index]
# Plot the original signal and its Fourier Transform
plt.figure(figsize=(10, 6))

# Plot the original signal 
plt.subplot(2, 1, 1)
plt.plot(data_Gyr['Time (s)'],data_Gyr["Gyroscope z (rad/s)"])
plt.title('The original signal of the Z-axis gyroscope reads at 900rpm with the engine running but the vehicle stationary')
plt.xlabel('Time')
plt.ylabel('Amplitude (rad/s)')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(frequencies,2* np.abs(fft_result_Gyr_z)/len(fft_result_Gyr_z))
plt.title('Fourier Transform of Z-axis Gyroscope')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  # Display only positive frequencies
plt.tight_layout()

plt.savefig("plot/Gyroscope/fft_z_900rpm.png")
plt.show()
print(max_amplitude_frequency)

#---------------------calculate the Fourier Trasform of x-axis Gyroscopo-----------

fft_result_Gyr_x = np.fft.fft(data_Gyr["Gyroscope x (rad/s)"])

# Plot the original signal and its Fourier Transform
plt.figure(figsize=(10, 6))

# Plot the original signal 
plt.subplot(2, 1, 1)
plt.plot(data_Gyr['Time (s)'],data_Gyr["Gyroscope x (rad/s)"])
plt.title('The original signal of the X-axis gyroscope reads at 900rpm',fontsize=16)
plt.xlabel('Time')
plt.ylabel('Amplitude (rad/s)')
plt.xticks(fontsize=12)  # Adjust the fontsize as needed
plt.yticks(fontsize=12)

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(frequencies, 2*np.abs(fft_result_Gyr_x)/len(fft_result_Gyr_x))
plt.title('Fourier Transform of X-axis Gyroscope at 900 rpm',fontsize=16)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (rad/s)')
plt.xlim(0, sampling_rate / 2)
plt.xticks(fontsize=12)  # Adjust the fontsize as needed
plt.yticks(fontsize=12)  # Display only positive frequencies
plt.tight_layout()
plt.savefig("plot/Gyroscope/fft_x_900rpm.png")
plt.show()