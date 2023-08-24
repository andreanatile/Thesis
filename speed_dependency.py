import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def filtered_signal(S,cutoff_frequency,sampling_frequency,order):
    # Define the Butterworth high-pass filter
    b, a = signal.butter(order, cutoff_frequency / (0.5 * sampling_frequency), btype='high')

    # Apply the high-pass filter to the accelerometer signal
    filtered_accel_signal = signal.filtfilt(b, a, S)
    return filtered_accel_signal

def Speed_Dependency(cutoff_frequency,sampling_frequency,order,S):
    filtered_accel_signal=filtered_signal(S,cutoff_frequency,sampling_frequency,order)
    # Calculate the moving average filter E with a rolling window
    rolling_window_size = 2000

    #E = data["Acceleration y (m/s^2)"].rolling(window=rolling_window_size, min_periods=1, center=False).mean()
    filtered_accel_series = pd.Series(filtered_accel_signal)
    E=np.abs(filtered_accel_series).rolling(window=rolling_window_size, min_periods=1, center=False).mean()
    # Calculate H ° S(t) and E ° |H| ° Xacc(t)
    H_composed = filtered_accel_signal  # H ° S(t)
    E_composed = E  # E ° |H| ° Xacc(t)

    dS= H_composed / E_composed
    return dS
