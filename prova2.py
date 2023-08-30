import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal,Speed_Linear_Regression
from featrures_extraction import features_extraction,SWT_Sym5,All_Features
import pywt

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
Sgyr=pd.read_csv("data/smalldrive/Gyroscope.csv")
Sgyr=Sgyr.iloc[:-1]
Location=pd.read_csv("data/smalldrive/Location.csv")
Location=Location.dropna()

pd.set_option('display.max_columns', None)
print(Location.head())
interpolet_speed=np.interp(Sacc['Time (s)'],Location['Time (s)'],Location['Velocity (m/s)'])

gyr_inter=Speed_Linear_Regression(interpolet_speed,Sgyr['Gyroscope x (rad/s)'])
dXgyr=Speed_Dependency(30,400,4,Sgyr['Gyroscope x (rad/s)'],Sacc['Acceleration y (m/s^2)'])
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(Sgyr['Time (s)'],Sgyr['Gyroscope x (rad/s)'])
plt.title('raw signal')
plt.xlabel('Time')
plt.ylabel('m/s2')

plt.subplot(3, 1, 2)
plt.plot(Sgyr['Time (s)'],gyr_inter)
plt.title('linear regression')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(Sgyr['Time (s)'],dXgyr)
plt.title('demodulated')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
