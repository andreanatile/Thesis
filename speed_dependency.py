import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import linregress

def filtered_signal(S,cutoff_frequency,sampling_frequency,order):
    # Define the Butterworth high-pass filter
    b, a = signal.butter(order, cutoff_frequency / (0.5 * sampling_frequency), btype='high')

    # Apply the high-pass filter to the accelerometer signal
    filtered_S_signal = signal.filtfilt(b, a, S)
    return filtered_S_signal

def Speed_Dependency(cutoff_frequency,sampling_frequency,order,S,Yacc):
    filtered_S_signal=filtered_signal(S,cutoff_frequency,sampling_frequency,order)
    # Calculate the moving average filter E with a rolling window
    rolling_window_size = 2000

    filtered_Yacc=filtered_signal(Yacc,30,400,4)
    filtered_Yacc_series_ = pd.Series(filtered_Yacc)
    # Calculate H ° S(t) and E ° |H| ° Yacc(t)
    E=np.abs(filtered_Yacc_series_).rolling(window=rolling_window_size, min_periods=1, center=False).mean()
    
    H_composed =filtered_S_signal   # H ° S(t)
    E_composed = E  # E ° |H| ° Yacc(t)

    dS= H_composed / E_composed
    return dS


from scipy.stats import linregress

def Speed_Linear_Regression(Velocity,Acceleration):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(Velocity, Acceleration)

    # Calculate residuals
    predicted_vertical_acceleration = slope * Velocity + intercept
    residuals = Acceleration - predicted_vertical_acceleration
    return residuals

