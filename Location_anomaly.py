import numpy as np
import pandas as pd

merge08=pd.read_csv('data/merge08.csv')
merge11=pd.read_csv('data/merge08.csv')

columns=['Time Start (s)', 'Time End (s)','Anomaly']
merge08=merge08[columns]
merge11=merge11[columns]

Location08=pd.read_csv('data/2023-09-08/Location.csv')
Location11=pd.read_csv('data/2023-09-08/Location.csv')
print(merge08.head())
merged=merge08.merge(Location08,left_on="Time Start (s)",right_on='Time (s)',how='inner')

print(len(merged))