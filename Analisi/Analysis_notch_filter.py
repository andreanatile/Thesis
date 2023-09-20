import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

Sacc900=pd.read_csv("data/fermo 900rpm/Accelerometer.csv")
Sacc1500=pd.read_csv("data/fermo 1500rpm/Accelerometer.csv")
Sacc2000=pd.read_csv("data/fermo 2000rpm/Accelerometer.csv")
Sgyr2000=pd.read_csv("data/fermo 2000rpm/Gyroscope.csv")

#filter the start and the end of the measurement for removing the touch of the screen
Sacc900 = Sacc900[(Sacc900['Time (s)'] > 1) & (Sacc900['Time (s)'] < 10)]
Sacc1500 = Sacc1500[(Sacc1500['Time (s)'] > 1) & (Sacc1500['Time (s)'] < 10)]
Sacc2000 = Sacc2000[(Sacc2000['Time (s)'] > 1) & (Sacc2000['Time (s)'] < 10)]
Sgyr2000 = Sgyr2000[(Sgyr2000['Time (s)'] > 2) & (Sgyr2000['Time (s)'] < 9)]
sampling_rate=400
#----------------Fourier Trasform 900rpm---------------------------------

#calculate value of FFT and frequency
Sacc900_noOffset=Sacc900['Acceleration y (m/s^2)']-np.mean(Sacc900['Acceleration y (m/s^2)'])
fft_900=np.fft.fft(Sacc900_noOffset)
freq_900=np.fft.fftfreq(len(Sacc900['Acceleration y (m/s^2)']),d=1/sampling_rate)


# Plot the original signal at 900rpm
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Sacc900['Time (s)'],Sacc900["Acceleration y (m/s^2)"])
plt.title('Original Signal of Acceleration y at 900rpm')
plt.xlabel('Time')
plt.ylabel('Amplitude (m/s^2)')

# Plot the Fourier Transform at 900rpm
plt.subplot(2, 1, 2)
plt.plot(freq_900, np.abs(fft_900))
plt.title('Fourier Transform of Acceleration y at 900rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

# Save and plot the graph
plt.savefig("plot/Accelerometer/fft_900rpm.png")
plt.show()


#---------------------------Fourier trasform at 2000rpm--------------------------------
Sacc2000_noOffset=Sacc2000['Acceleration y (m/s^2)']-np.mean(Sacc2000['Acceleration y (m/s^2)'])
fft_2000=np.fft.fft(Sacc2000_noOffset)
freq_2000=np.fft.fftfreq(len(Sacc2000['Acceleration y (m/s^2)']),d=1/sampling_rate)


# Plot the original signal (optional)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Sacc2000['Time (s)'],Sacc2000["Acceleration y (m/s^2)"])
plt.title('Original Signal of Acceleration y at 2000rpm')
plt.xlabel('Time')
plt.ylabel('Amplitude (m/s^2)')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(freq_2000, np.abs(fft_2000))
plt.title('Fourier Transform of Acceleration y at 2000rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

# Save and plot the graph
plt.savefig("plot/Accelerometer/fft_2000rpm.png")
plt.show()


#-----------------------------FFT at 1500rpm----------------------------------------
Sacc1500_noOffset=Sacc1500['Acceleration y (m/s^2)']-np.mean(Sacc1500['Acceleration y (m/s^2)'])
fft_1500=np.fft.fft(Sacc1500_noOffset)
freq_1500=np.fft.fftfreq(len(Sacc1500['Acceleration y (m/s^2)']),d=1/sampling_rate)
# Plot the original signal (optional)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Sacc1500['Time (s)'],Sacc1500["Acceleration y (m/s^2)"])
plt.title('Original Signal of Acceleration y at 1500rpm')
plt.xlabel('Time')
plt.ylabel('Amplitude (m/s^2)')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(freq_1500, np.abs(fft_1500))
plt.title('Fourier Transform of Acceleration y at 1500rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

# Save and plot the graph
plt.savefig("plot/Accelerometer/fft_1500rpm.png")
plt.show()


#-----------------------------confronting the two FFT----------------------------

plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(freq_900,np.abs(fft_900))
plt.title('Fourier Transform of Acceleration y at 900rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2) 

plt.subplot(3,1,2)
plt.plot(freq_1500,np.abs(fft_1500))
plt.title('Fourier Transform of Acceleration y at 1500rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)

plt.subplot(3,1,3)
plt.plot(freq_2000,np.abs(fft_2000))
plt.title('Fourier Transform of Acceleration y at 2000rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)
plt.tight_layout()

# Save and plot the graph
plt.savefig("plot/Accelerometer/fft_900_vs_1500_vs_2000.png")
plt.show()

# Finding why we have a wide band of noise at 1500rpm (becouse the accelerator pedal wasn't firm)
# Parameters for spectrogram
nperseg = 256       # Number of data points per segment
noverlap = 128      # Number of overlapping data points between segments

# Calculate the spectrogram of Acceleration y at 1500rpm
freqs, times, Sxx = spectrogram(Sacc1500["Acceleration y (m/s^2)"], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, freqs, 10 * np.log10(Sxx), shading='gouraud')  # Convert to dB scale
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram of Accelerometer y-axis Data')
plt.ylim(0, sampling_rate / 2)  # Display only positive frequencies
plt.tight_layout()

plt.savefig("plot/Accelerometer/spectogram_1500rpm.png")
plt.show()

#----------------------------- FFT gyroscope x at 2000rpm----------------------------
fft_2000=np.fft.fft(Sgyr2000['Gyroscope x (rad/s)'])
freq_2000=np.fft.fftfreq(len(Sgyr2000['Gyroscope x (rad/s)']),d=1/sampling_rate)


# Plot the original signal (optional)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(Sgyr2000['Time (s)'],Sgyr2000['Gyroscope x (rad/s)'])
plt.title('Original Signal of Gyroscope x at 2000rpm')
plt.xlabel('Time')
plt.ylabel('Amplitude (rad/s)')

# Plot the Fourier Transform
plt.subplot(2, 1, 2)
plt.plot(freq_2000, np.abs(fft_2000))
plt.title('Fourier Transform of Gyroscope x at 2000rpm')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude  (rad/s)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.savefig("plot/Gyroscope/fft_200rpm.png")
plt.show()
