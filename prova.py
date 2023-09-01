import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
location=pd.read_csv('data/smalldrive/Location.csv')


dYacc=Speed_Dependency(30,400,4,Sacc['Acceleration y (m/s^2)'],Sacc['Acceleration y (m/s^2)'])
filtered_y_signal=filtered_signal(Sacc['Acceleration y (m/s^2)'],30,400,4)
# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(Sacc['Time (s)'],dYacc, label='dyacc')
plt.plot(Sacc['Time (s)'],filtered_y_signal,label="filtered signal")
plt.plot(location['Time (s)'], location['Velocity (m/s)'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Comparison of Unfiltered dyacc and Filtered Data')
plt.grid(True)
plt.show()

Sgyr=pd.read_csv("data/smalldrive/Gyroscope.csv")
Sacc=Sacc.iloc[:-1]
print(len(Sgyr))
print(len(Sacc))
dXgyr=Speed_Dependency(30,400,4,Sgyr['Gyroscope x (rad/s)'],Sacc['Acceleration y (m/s^2)'])
filtered_x_Gyr=filtered_signal(Sgyr['Gyroscope x (rad/s)'],30,400,4)


plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(Sgyr['Time (s)'],filtered_x_Gyr)
plt.title('filtered signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(Sgyr['Time (s)'],Sgyr['Gyroscope x (rad/s)'])
plt.title('raw signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3,1,3)
plt.plot(Sgyr['Time (s)'],dXgyr)
plt.title('dXGyr')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

