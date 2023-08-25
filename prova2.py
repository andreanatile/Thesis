import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from speed_dependency import Speed_Dependency,filtered_signal
from featrures_extraction import features_extraction
import pywt

Sacc=pd.read_csv("data/smalldrive/Accelerometer.csv")

# Example accelerometer data (replace with your data)
accel_data = Sacc['Acceleration y (m/s^2)'].iloc[:1000]

# Define wavelet and decomposition level
wavelet = 'sym5'
level = 3

# Perform stationary wavelet transform
coeffs = pywt.swt(accel_data, wavelet, level=level)

# Extract approximation and detail coefficients from the result
approx_coeffs, detail_coeffs = zip(*coeffs)

print(type(detail_coeffs))