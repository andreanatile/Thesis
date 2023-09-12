import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from scipy.signal import butter, filtfilt

Sacc=pd.read_csv("data/09-08-2023/Accelerometer.csv")

plt.figure(figsize=(10,6))
plt.plot(Sacc['Time (s)'],Sacc["Acceleration y (m/s^2)"])
plt.title('Original Signal of Acceleration y (m/s^2)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()
