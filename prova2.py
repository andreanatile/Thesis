import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal,Speed_Linear_Regression
from featrures_extraction import features_extraction,SWT_Sym5,All_Features,UpSampling
import pywt

Sacc=pd.read_csv("data/longdrive/Accelerometer.csv")
Sgyr=pd.read_csv("data/longdrive/Gyroscope.csv")
pd.set_option('display.max_columns', None)

upsampled_Sacc=UpSampling(Sacc['Acceleration y (m/s^2)'],100,400)
upsampled_Time=UpSampling(Sacc['Time (s)'],100,400)
upsampled_GyrX=UpSampling(Sgyr['Gyroscope x (rad/s)'],100,400)
upsampled_GyrZ=UpSampling(Sgyr['Gyroscope z (rad/s)'],100,400)

features=All_Features(upsampled_Sacc,upsampled_GyrZ,upsampled_GyrZ,upsampled_Time,1000,0.66,400,3,['Acceleration y','Gyroscope x','Gyroscope z'])
print(len(features))
#Even with a 400Hz of sampling frequency over 21 minutes of road measuament i can 
# analyze data in a small period of time.
print(features.tail())