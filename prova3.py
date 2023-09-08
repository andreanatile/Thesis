import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
import featrures_extraction as ft
from notch_filter import notch_filter


Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
location=pd.read_csv('data/smalldrive/Location.csv')
SaccY=Sacc['Acceleration y (m/s^2)']

segments=ft.Segmentation(SaccY,1000,0.66)

sampling_rate = 400
flag=0
FFT_Segments=[]
filtered_Segments=[]
for segment in segments:
    FFT_Segments.append(np.fft.fft(segment))

for i in range(0,len(FFT_Segments)):
    if i==0:
        previous_max_freq_index=pervious_max_amplitude=0
        max_freq_index,max_energy,b,a=notch_filter(FFT_Segments[0],previous_max_freq_index,pervious_max_amplitude,sampling_rate,0.66)
        previous_max_freq_index,pervious_max_amplitude=max_freq_index,max_energy
        if b is not None:
            filtered_Segments.append(signal.filtfilt(b,a,segments[i]))
        else:
            filtered_Segments.append(segments[i])
    else:
        
        max_freq_index,max_energy,b,a=notch_filter(FFT_Segments[0],previous_max_freq_index,pervious_max_amplitude,sampling_rate,0.66)
        previous_max_freq_index,pervious_max_amplitude=max_freq_index,max_energy
        if b is not None:
            flag+=1
            filtered_Segments.append(signal.filtfilt(b,a,segments[i]))
        else:
            filtered_Segments.append(segments[i])


print(len(segments))
print(flag)

 
