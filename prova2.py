import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction
import pywt

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")

arr=[[1,2,34],[12,3,4],[1,5,6]]
print(arr[2][2])

features=features_extraction(Sacc['Acceleration y (m/s^2)'],1000,0.66,400,3)

print(features.head())