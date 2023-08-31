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
features=All_Features(Sacc['Acceleration y (m/s^2)'],Sgyr['Gyroscope x (rad/s)'],Sgyr['Gyroscope z (rad/s)'],Sacc['Time (s)'],1000,0.66,400,3,['Acceleration Y','Gyroscope X','Gyroscope Z'])
pd.set_option('display.max_columns', None)

print(features.columns)