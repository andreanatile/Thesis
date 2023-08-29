import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction,SWT_Sym5,All_Features
import pywt

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")
Sgyr=pd.read_csv("data/smalldrive/Gyroscope.csv")

""" features=features_extraction(Sacc['Acceleration y (m/s^2)'],1000,0.66,400,3)
features.columns=features.columns + ' Acceleration y' """
pd.set_option('display.max_columns', None)

SaccY=Sacc['Acceleration y (m/s^2)']
SgyrX=Sgyr["Gyroscope x (rad/s)"]
SgyrZ=Sgyr["Gyroscope z (rad/s)"]
all_features=All_Features(SaccY,SgyrX,SgyrZ,1000,0.66,400,3,[" Acceleration y"," Gyroscope x"," Gyroscope z"])
print(all_features.head())