import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pywt

def features_extraction(data,segment_length,overlap_percentage,sampling_frequency,level):
    overlap_length = int(segment_length * overlap_percentage)  # Length of the overlap
    # Initialize an empty list to store the segments
    segments = []
    #acceleration_signal=data['Acceleration y (m/s^2)']

    # Create segments as before
    for i in range(0, len(data) - segment_length + 1, segment_length - overlap_length):
        segment = data[i:i+segment_length]
        segments.append(segment)

    # Create a DataFrame from the segments, using np array for efficency
    num_segments=len(segments)
    #time domain features
    N_segment=[]
    Mean=[]
    Std=[]
    Var=[]
    Ptp=[]
    Rms=[]
    Zcr=[]
    Mean_Abs=[]
    sma=[]
    mean_frequency=[]
    median_frequency=[]
    approx_absolute_mean_wv=[]
    detail_absolute_mean_wv=[]
    approx_std_wv=[]
    detail_std_wv=[]
    approx_var_wv=[]
    detail_var_wv=[]
    approx_energy_wv=[]
    detail_energy_wv=[]

    for i in range(0,len(segments)):
        N_segment.append(i)
        Mean.append(segments[i].mean())
        Std.append(segments[i].std())
        Var.append(segments[i].var())
        Ptp.append(np.ptp(segments[i]))
        Rms.append(np.sqrt(np.mean(segments[i]**2)))
        Zcr.append(np.sum(np.diff(np.sign(segments[i])) != 0) / (2 * len(segments[i])))
        Mean_Abs.append(np.abs(segments[i]).mean())
        sma.append(np.sum(abs(segments[i]))) #not sure of abs or not, wikipedia say not
        #frequency domain
        ham_ftt,freq_axis=Hamming_Window_FFT(segments[i],sampling_frequency)
        mean_frequency.append(Mean_Frequency(ham_ftt,freq_axis))
        median_frequency.append(Median_Frequency(ham_ftt,freq_axis))
        #wavelet domain
        approx_coeffs,detail_coeffs=SWT_Sym5_Level4(segments[i])
        #absolute mean of approx and detail
        a_abs_mean,d_abs_mean=Absolute_Mean_WV(approx_coeffs,detail_coeffs,level)
        approx_absolute_mean_wv.append(a_abs_mean)
        detail_absolute_mean_wv.append(d_abs_mean)
        #standard deviation of approx and detail
        a_std,d_std=Std_WV(approx_coeffs,detail_coeffs)
        approx_std_wv.append(a_std)
        detail_std_wv.append(d_std)
        #variance 
        a_var,d_var=Var_WV(approx_coeffs,detail_coeffs)
        approx_var_wv.append(a_var)
        detail_var_wv.append(d_var)
        #energy of every layer
        a_en,d_en=Energy_WV(approx_coeffs,detail_coeffs)
        approx_energy_wv.append(a_en)
        detail_energy_wv.append(d_en)

    features=pd.DataFrame({'N_segment': N_segment,
                           'Mean':Mean,
                           'Standard deviation':Std,
                           'Variance':Var,
                           'Peak to peak':Ptp,
                           'Root mean square':Rms,
                           'Zero crossing rate':Zcr,
                           'Mean absolute':Mean_Abs,
                           'Signal magnitude area':sma,
                           'Mean frequency':mean_frequency,
                           'Median frequency':median_frequency})
    #label_abs_mean=['a1 absolute mean','a2 absolute mean','a3 absolute mean','d1 absolute mean','d2 absolute mean','d3 absolute mean']
    abs_df=Create_WV_Datasets(approx_absolute_mean_wv,detail_absolute_mean_wv,['a1 absolute mean','a2 absolute mean','a3 absolute mean',
                                                                               'd1 absolute mean','d2 absolute mean','d3 absolute mean'])
    std_df=Create_WV_Datasets(approx_std_wv,detail_std_wv,['a1 std','a2 std','a3 std',
                                                           'd1 std','d2 std','d3 std'])
    var_df=Create_WV_Datasets(approx_var_wv,detail_var_wv,['a1 var','a2 var','a3 var',
                                                           'd1 var','d2 var','d3 var'])
    energy_df=Create_WV_Datasets(approx_energy_wv,detail_energy_wv,['a1 energy','a2 energy','a3 energy',
                                                           'd1 energy','d2 energy','d3 energy'])
    features=pd.concat([features,abs_df,std_df,var_df,energy_df])
    return features

def Hamming_Window_FFT(segment,sampling_frequency):
    ham=np.hamming(len(segment))
    segment_ham=segment*ham
    # Calculate FFT of the Hamming windowed signal
    ham_fft = np.abs(np.fft.fft(segment_ham))
    ham_fft = ham_fft[:len(segment) // 2]  # Discard half points
    # Generate frequency axis 
    freq_axis = np.fft.fftfreq(len(segment), 1/sampling_frequency)[:len(segment) // 2]
    return ham_fft,freq_axis

def Mean_Frequency(ham_fft,freq_axis):
    mean_frequency = np.sum(ham_fft * freq_axis) / np.sum(ham_fft)
    return mean_frequency

def Median_Frequency(ham_ftt,freq_axis):
    # Sort the FFT magnitudes in ascending order
    sorted_magnitudes = np.sort(ham_ftt)

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(sorted_magnitudes) / np.sum(sorted_magnitudes)

   # Find the index where CDF is closest to 0.5
    median_frequency_index = np.argmin(np.abs(cdf - 0.5))

    # Get the corresponding frequency from the freq_axis
    median_frequency = freq_axis[median_frequency_index]
    return median_frequency

def SWT_Sym5_Level4(segment):
    wavelet = 'sym5'
    level = 3

    # Perform stationary wavelet transform
    coeffs = pywt.swt(segment, wavelet, level=level)

    # Extract approximation and detail coefficients from the result
    approx_coeffs, detail_coeffs = zip(*coeffs)
    return approx_coeffs,detail_coeffs

def Absolute_Mean_WV(approx_coeffs,detail_coeffs,level):
    approx_abs_mean=np.zeros(level)
    detail_abs_mean=np.zeros(level)
    for i in range(len(approx_coeffs)):
        approx_abs_mean[i]=np.mean(np.abs(approx_coeffs[i]))
        detail_abs_mean[i]=np.mean(np.abs(detail_coeffs[i]))
    
    return approx_abs_mean,detail_abs_mean

def Std_WV(approx_coeffs,detail_coeffs):
    approx_std=[]
    detail_std=[]
    for i in range(len(approx_coeffs)):
        approx_std.append(np.std(approx_coeffs[i]))
        detail_std.append(np.std(detail_coeffs[i]))
    
    return approx_std,detail_std

def Var_WV(approx_coeffs,detail_coeffs):
    approx_var=[]
    detail_var=[]
    for i in range(len(approx_coeffs)):
        approx_var.append(np.var(approx_coeffs[i]))
        detail_var.append(np.var(detail_coeffs[i]))
    
    return approx_var,detail_var

def Energy_WV(approx_coeffs,detail_coeffs):
    approx_energy=[]
    detail_energy=[]
    for i in range(len(approx_coeffs)):
        approx_energy.append(np.sum(approx_coeffs[i]**2))
        detail_energy.append(np.sum(detail_coeffs[i]**2))
    
    return approx_energy,detail_energy

def Create_WV_Datasets(a,d,label):
    data1=pd.DataFrame(a,columns=label[:3])
    data2=pd.DataFrame(d,columns=label[3:])
    data=pd.concat([data1,data2])
    return data

def Wave_length(segment):
    segment_length = len(segment)
    if segment_length > 1:
        segment_waveform_length = sum(abs(segment[j] - segment[j-1]) for j in range(1, segment_length))
    else:
        segment_waveform_length=0
    return segment_waveform_length
