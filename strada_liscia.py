from notch_filter import notch_filtering
from featrures_extraction import Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

segment_length=1000
sampling_rate=400
overlap_percentage=0.66
Sacc=pd.read_csv("data/strada liscia/Accelerometer.csv")
Yacc=Sacc['Acceleration y (m/s^2)']


Yacc_segments=Segmentation(Yacc,segment_length,overlap_percentage)
freq=np.fft.fftfreq(1000,d=1/sampling_rate)
Yacc_filtered=notch_filtering(Yacc,segment_length,sampling_rate,overlap_percentage)

Yacc_FFT_segments=[]
Yacc_FFT_filtered_segments=[]
for i in range(0,len(Yacc_segments)):
    Yacc_segments[i]=Yacc_segments[i]-np.mean(Yacc_segments[i])
    Yacc_filtered[i]=Yacc_filtered[i]-np.mean(Yacc_filtered[i])
    Yacc_FFT_segments.append(np.fft.fft(Yacc_segments[i]))
    Yacc_FFT_filtered_segments.append(np.fft.fft(Yacc_filtered[i]))

print(len(Yacc_FFT_segments))
#Plot
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(freq,np.abs(Yacc_FFT_segments[28]))
plt.title("FFT raw segments")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(freq,np.abs(Yacc_FFT_filtered_segments[28]))
plt.title("FFT filtered segments")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()

plt.subplot(3,1,3)
plt.plot(freq,np.abs(Yacc_FFT_segments[21]))
plt.title("FFT  previuos raw segments")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (m/s^2)')
plt.xlim(0, sampling_rate / 2)  
plt.tight_layout()
plt.show()

