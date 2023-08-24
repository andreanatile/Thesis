import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def features_extraction(data,segment_length,overlap_percentage):
    overlap_length = int(segment_length * overlap_percentage)  # Length of the overlap
    # Initialize an empty list to store the segments
    segments = []
    #acceleration_signal=data['Acceleration y (m/s^2)']

    # Create segments as before
    for i in range(0, len(data) - segment_length + 1, segment_length - overlap_length):
        segment = data[i:i+segment_length]
        segments.append(segment)

    # Create a DataFrame from the segments
    #segment_df = pd.DataFrame(segments)
    
    #time domain features
    N_segment=[]
    Mean=[]
    Std=[]
    Var=[]
    Ptp=[]
    Rms=[]
    Zcr=[]
    Mean_Abs=[]
    waveform_length=[]
    sma=[]

    #frequency domain feature

    for i in range(0,len(segments)):
        N_segment.append(i)
        Mean.append(segments[i].mean())
        Std.append(segments[i].std())
        Var.append(segments[i].var())
        Ptp.append(segments[i].ptp())
        for segment in segments:
            rms = np.sqrt(np.mean(np.array(segment) ** 2))
            Rms.append(rms)
        Zcr.append(np.sum(np.diff(np.sign(segments[i])) != 0) / (2 * len(segments[i])))
        Mean_Abs.append(np.abs(segments[i]).mean())
        waveform_length.append(sum(abs(segments[i][j] - segments[i][j-1]) for j in range(1, len(segment[i]))))
        sma.append(np.sum(abs(segments[i]))) #not sure of abs or not, wikipedia say not

    features=pd.DataFrame({'N_segment': N_segment,
                           'Mean':Mean,
                           'Standard deviation':Std,
                           'Variance':Var,
                           'Peak to peak':Ptp,
                           'Root mean square':Rms,
                           'Zero crossing rate':Zcr,
                           'Mean absolute':Mean_Abs,
                           'waveform length':waveform_length,
                           'Signal magnitude area':sma})
    return features