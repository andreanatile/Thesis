from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
# Create/view notch filter
samp_freq = 400 # Sample frequency (Hz)
notch_freq = 70.0 # Frequency to be removed from signal (Hz)
band_size=10
quality_factor = notch_freq/band_size # Quality factor

# Design a notch filter using signal.iirnotch
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

# Compute magnitude response of the designed filter
freq, h = signal.freqz(b_notch, a_notch, fs=2*np.pi)

fig = plt.figure(figsize=(8, 6))

# Plot magnitude response of the filter
plt.plot(freq*samp_freq/(2*np.pi), 20 * np.log10(abs(h)),
		'r', label='Bandpass filter', linewidth='2')

plt.xlabel('Frequency [Hz]', fontsize=20)
plt.ylabel('Magnitude [dB]', fontsize=20)
plt.title('Notch Filter', fontsize=20)
plt.grid()
plt.show()
