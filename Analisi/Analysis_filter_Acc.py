import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

# Replace 'data.csv' with the actual path to your CSV file
csv_file_path = 'data/fermo/Accelerometer.csv'

# Read the CSV file into a DataFrame
Sacc = pd.read_csv(csv_file_path)
Sacc=Sacc[Sacc['Time (s)']<5]
pd.set_option('display.max_columns', None)
#print(data.head())


# Calculate the Fourier Transform of accelerometer y axis
fft_result = np.fft.fft(Sacc["Acceleration y (m/s^2)"])

# Calculate the corresponding frequency values
sampling_rate = 400
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
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.show()
print(max_amplitude_frequency)

# Parameters for spectrogram
nperseg = 256       # Number of data points per segment
noverlap = 128      # Number of overlapping data points between segments

# Calculate the spectrogram of Acceleration y 
freqs, times, Sxx = spectrogram(Sacc["Acceleration y (m/s^2)"], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')  # Convert to dB scale
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Accelerometer y-axis Data')
plt.ylim(0, sampling_rate / 2)  # Display only positive frequencies
plt.tight_layout()
plt.show()

# Define the Butterworth high-pass filter parameters
order = 4  # Filter order
cutoff_freq = 30  # Cutoff frequency in Hz

# Compute the Nyquist frequency (half of the sampling frequency)
nyquist_freq = 0.5 / (Sacc['Time (s)'].iloc[1] - Sacc['Time (s)'].iloc[0])
normalized_cutoff = cutoff_freq / nyquist_freq

# Design the Butterworth filter
b, a = butter(order, normalized_cutoff, btype='high', analog=False)

# Apply the filter to the xacc data column
filtered_yacc = filtfilt(b, a, Sacc["Acceleration y (m/s^2)"])

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(Sacc['Time (s)'], Sacc["Acceleration y (m/s^2)"], label='Original yacc')
plt.plot(Sacc['Time (s)'], filtered_yacc, label='Filtered yacc')
plt.xlabel('Time (s)')
plt.ylabel('yacc')
plt.legend()
plt.title('Original vs Filtered yacc')
plt.grid(True)
plt.show()


# Remove the DC ofstet, fisrt calculate the mean value

mean_value=np.mean(Sacc['Acceleration y (m/s^2)'])
Sacc_no_ofset=Sacc['Acceleration y (m/s^2)']-mean_value

# Plot the FFT transform

fft_no_ofset=np.fft.fft(Sacc_no_ofset)

plt.figure(figsize=(10,6))

# Plot FFT with no ofset
plt.subplot(2,1,1)
plt.plot(frequencies,np.abs(fft_no_ofset))
plt.title('Fourier Transform of Acceleration y (m/s^2) with no ofset')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

#Plot raw fft
plt.subplot(2,1,2)
plt.plot(frequencies,np.abs(fft_result))
plt.title('Fourier Transform of Acceleration y (m/s^2) with  ofset')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.show()


# Plot fft with no ofset  with amplitude in dB vs no dB
# Plot FFT dB
plt.subplot(2,1,1)
plt.plot(frequencies,20*np.log10(np.abs(fft_no_ofset)))
plt.title('Fourier Transform of Acceleration y (m/s^2) with no ofset in dB')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude in dB')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

#Plot raw fft
plt.subplot(2,1,2)
plt.plot(frequencies,np.abs(fft_no_ofset))
plt.title('Fourier Transform of Acceleration y (m/s^2) with no ofset')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.show()

