import pandas as pd
import labelling as lb
import numpy as np
import featrures_extraction as ft



pd.set_option('display.max_columns', None)
segment_length=1000
overlap_percentage=0.66
sampling_rate=400
level=3
suffix=['Acceleration y','Gyroscope x','Gyroscope z']
# Read data from sensor
Sacc11=pd.read_csv("data/2023-09-11/Accelerometer.csv")
Sgyr11=pd.read_csv("data/2023-09-11/Gyroscope.csv")

# Save the data from axis of interest
Yacc11=Sacc11['Acceleration y (m/s^2)']
Xgyr11=Sgyr11['Gyroscope x (rad/s)']
Zgyr11=Sgyr11['Gyroscope z (rad/s)']
Time11=Sacc11['Time (s)']

#------------------------------second data measurement-----------------------------
# Read data from sensor
Sacc08=pd.read_csv("data/2023-09-08/Accelerometer.csv")
Sgyr08=pd.read_csv("data/2023-09-08/Gyroscope.csv")

# Save the data from axis of interest
Yacc08=Sacc08['Acceleration y (m/s^2)']
Xgyr08=Sgyr08['Gyroscope x (rad/s)']
Zgyr08=Sgyr08['Gyroscope z (rad/s)']
Time08=Sacc08['Time (s)']

# Segmenting the data in order to apply notch filter
list_signals=[Yacc11,Xgyr11,Zgyr11,Yacc08,Xgyr08,Zgyr08]
list_segments=[]

# Append in the list all the segments for every signal
for signal in list_signals:
    list_segments.append(ft.Segmentation(signal,segment_length,overlap_percentage))

# Calculate the combining features from the two measurement using segments
features11=ft.Combining_features_from_segments(list_segments[0],list_segments[1],
                                               list_segments[2],Time11,segment_length,
                                               overlap_percentage,sampling_rate,
                                               suffix)

features08=ft.Combining_features_from_segments(list_segments[3],list_segments[4],
                                               list_segments[5],Time08,segment_length,
                                               overlap_percentage,sampling_rate,
                                               suffix)

# Read the lables for the measurement


with open('data/2023-09-08/labels.txt', 'r') as file: # for 2023-09-08
    # Read the entire content of the file into a string
    file_contents08 = file.read()
# Extract labels
labels08 = lb.Extract_Labels_fromTeroSubliter(file_contents08)
merge08 = lb.Merge_Feature_Label(features08, labels08)

with open('data/2023-09-11/labels.txt', 'r') as file: # for 2023-09-11
    # Read the entire content of the file into a string
    file_contents11 = file.read()
# Extract labels
labels11 = lb.Extract_Labels_fromTeroSubliter(file_contents11)
merge11 = lb.Merge_Feature_Label(features11, labels11)

# Drop the time columns since we won't use them anymore
merge11 = merge11.drop(['Time Start (s)', 'Time End (s)'], axis=1)
merge08 = merge08.drop(['Time Start (s)', 'Time End (s)'], axis=1)

# Concat the two dataset, creating the dataset for training the ML algorithm
Training_Dataset = pd.concat([merge11, merge08], ignore_index=True)
Training_Dataset['Anomaly'].replace(
    ['Mild', 'Severe','Span'], 'Anomaly', inplace=True) 
# Save as csv file
Training_Dataset.to_csv(
    "data/Training_Datasets/notchFiltered_Training_Dataset.csv", index=False)
print(len(Training_Dataset[Training_Dataset['Anomaly']=='Potholes']))
print(len(Training_Dataset[Training_Dataset['Anomaly']=='ok']))
print(len(Training_Dataset[Training_Dataset['Anomaly']=='Span']))