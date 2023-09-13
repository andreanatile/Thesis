import pandas as pd
import labelling as lb
import numpy as np
import featrures_extraction as ft
pd.set_option('display.max_columns', None)

# Read data from sensor
Sacc=pd.read_csv("data/2023-09-11/Accelerometer.csv")
Sgyr=pd.read_csv("data/2023-09-11/Gyroscope.csv")

# Save the data from axis of interest
Yacc=Sacc['Acceleration y (m/s^2)']
Xgyr=Sgyr['Gyroscope x (rad/s)']
Zgyr=Sgyr['Gyroscope z (rad/s)']
Time=Sacc['Time (s)']

# Extract feature
feature=ft.All_Features(Yacc,Xgyr,Zgyr,Time,1000,0.66,400,3,['Acceleration y','Gyroscope x','Gyroscope z'])
# Open the text file in read mode
with open('data/2023-09-11/labels.txt', 'r') as file:
    # Read the entire content of the file into a string
    file_contents = file.read()

labels=lb.Extract_Labels_fromTeroSubliter(file_contents)

merge=lb.Merge_Feature_Label(feature,labels)
print(merge.tail())



