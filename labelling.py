import pandas as pd
import re

def Extract_Labels_fromTeroSubliter(data_string):
    # Initialize empty lists to store extracted information
    number = []
    time_start = []
    time_end = []
    anomaly = []

    # Split the data using a regular expression to separate each entry
    entries = re.split(r'\n(?=\d+\n)', data_string.strip())



    # Loop through the entries and extract relevant information
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) == 3:
            num = int(lines[0])
            time_parts = re.findall(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[1])
            if time_parts:
                start_time = time_to_seconds(time_parts[0][0])
                end_time = time_to_seconds(time_parts[0][1])
                anomaly_text = lines[2]
                number.append(num)
                time_start.append(start_time)
                time_end.append(end_time)
                anomaly.append(anomaly_text)

    # Create a DataFrame
    df = pd.DataFrame({
            'Number': number,
        'Time Start (s)': time_start,
        'Time End (s)': time_end,
        'Anomaly': anomaly
    })
    
    # Replace NaN values in the Anomaly column with 'ok'
    df['Anomaly'].fillna('ok', inplace=True)

    # Return the dataframe
    return df

# Function to convert time format to seconds
def time_to_seconds(time_str):
    parts = time_str.split(':')
    seconds = float(parts[-1].replace(',', '.'))
    seconds += int(parts[-2]) * 60
    seconds += int(parts[-3]) * 3600
    return seconds

def Merge_Feature_Label(feature,labels):
    merged_df = feature.copy()  # Create a copy of df1 to keep the original intact
    for idx, row in labels.iterrows():
        mask = (merged_df['Time Start (s)'] <= row['Time Start (s)']) & (merged_df['Time End (s)'] >= row['Time End (s)'])
        if mask.any():
            merged_df.loc[mask, 'Anomaly'] = row['Anomaly']

    # Replace NaN values in the Anomaly column with 'ok'
    merged_df['Anomaly'].fillna('ok', inplace=True)
    
    return merged_df

