import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

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
    E=np.abs(filtered_Yacc_series_).rolling(window=rolling_window_size, min_periods=1, center=False).mean()
    # Calculate H ° S(t) and E ° |H| ° Xacc(t)
    H_composed =filtered_S_signal   # H ° S(t)
    E_composed = E  # E ° |H| ° Xacc(t)

    dS= H_composed / E_composed
    return dS

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Simulated data
horizontal_velocity = np.random.rand(100)  # Horizontal velocity
vertical_acceleration = 3 * horizontal_velocity + 0.5 * np.random.randn(100)  # Vertical acceleration with noise

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(horizontal_velocity, vertical_acceleration)

# Calculate residuals
predicted_vertical_acceleration = slope * horizontal_velocity + intercept
residuals = vertical_acceleration - predicted_vertical_acceleration

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(horizontal_velocity, vertical_acceleration, label='Original Data')
plt.plot(horizontal_velocity, predicted_vertical_acceleration, color='red', label='Regression Line')
plt.xlabel('Horizontal Velocity')
plt.ylabel('Vertical Acceleration')
plt.legend()
plt.title('Removing Horizontal Velocity Influence')
plt.show()
