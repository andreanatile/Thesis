import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
print(Sacc.head())
location=pd.read_csv('data/smalldrive/Location.csv')

dYacc=Speed_Dependency(30,400,4,Sacc['Acceleration y (m/s^2)'])
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
dYgyr=Speed_Dependency(30,400,4,Sgyr['Gyroscope y (rad/s)'])
filtered_y_Gyr=filtered_signal(Sgyr['Gyroscope y (rad/s)'],30,400,4)
plt.figure(figsize=(10, 6))
plt.plot(Sgyr['Time (s)'],dYgyr, label='dYgyr')
plt.plot(Sgyr['Time (s)'],filtered_y_Gyr,label="filtered signal")
plt.plot(Sgyr['Time (s)'],Sgyr['Gyroscope y (rad/s)'], label="raw signal")
plt.plot(location['Time (s)'], location['Velocity (m/s)'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Comparison of Unfiltered dyacc and Filtered Data')
plt.grid(True)
plt.show()


features=features_extraction(Sacc,1000,0.66)
print(features.head())