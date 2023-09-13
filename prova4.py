import pandas as pd



# Open the text file in read mode
with open('data/2023-09-11/labels.txt', 'r') as file:
    # Read the entire content of the file into a string
    file_contents = file.read()

# Now, file_contents contains the content of the file as a string
print(file_contents)
