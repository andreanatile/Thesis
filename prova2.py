import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction
import pywt

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")

features=features_extraction(Sacc['Acceleration y (m/s^2)'],1000,0.66,400,3)