import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

Sgyr2000=pd.read_csv("data/fermo 2000rpm/Gyroscope.csv")
Sgyr900=pd.read_csv('data/fermo 900rpm/Gyroscope.csv')

Sgyr2000 = Sgyr2000[(Sgyr2000['Time (s)'] > 2) & (Sgyr2000['Time (s)'] < 9)]
Sgyr900 = Sgyr900[(Sgyr900['Time (s)'] > 2) & (Sgyr900['Time (s)'] < 9)]
sampling_rate=400

fft_2000=np.fft.fft(Sgyr2000['Gyroscope x (rad/s)'])
fft_2000=2*(fft_2000-np.mean(fft_2000))/len(fft_2000)
fft_900=np.fft.fft(Sgyr900['Gyroscope x (rad/s)'])
fft_900=2*(fft_900-np.mean(fft_900))/len(fft_900)

freq_2000=np.fft.fftfreq(len(Sgyr2000['Gyroscope x (rad/s)']),d=1/sampling_rate)
freq_900=np.fft.fftfreq(len(Sgyr900['Gyroscope x (rad/s)']),d=1/sampling_rate)


# Plot the original signal (optional)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(freq_2000,np.abs(fft_900))
plt.title('Fourier Transform of Gyroscope x at 900 rpm',fontsize=16)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude  (rad/s)')
plt.xlim(0, sampling_rate / 2)  
plt.xticks(fontsize=12)  # Adjust the fontsize as needed
plt.yticks(fontsize=12)
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.plot(freq_2000,np.abs(fft_2000))
plt.title('Fourier Transform of Gyroscope x at 2000 rpm',fontsize=16)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude  (rad/s)')
plt.xlim(0, sampling_rate / 2)  
plt.xticks(fontsize=12)  # Adjust the fontsize as needed
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("plot/Gyroscope/fftx_900_vs_2000rpm.png")
plt.show()

