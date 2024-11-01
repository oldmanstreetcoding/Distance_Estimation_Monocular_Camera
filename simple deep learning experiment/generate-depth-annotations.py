'''
Purpose: Generate dataset for depth estimation.
This script filters relevant object data from the KITTI annotations, removes "DontCare" entries, 
extracts key features needed for depth estimation, and splits the data into training and testing datasets.
The result is saved in separate CSV files for easy access during model training and evaluation.
'''

import pandas as pd
from tqdm import tqdm
import os
import numpy as np

# Load the annotations CSV file generated from the KITTI dataset
df = pd.read_csv('annotations.csv')

# Filter out rows labeled as 'DontCare' to focus on relevant objects
new_df = df.loc[df['class'] != 'DontCare']

# Initialize an empty DataFrame to store the selected columns for depth estimation
result_df = pd.DataFrame(columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 
                                  'angle', 'xloc', 'yloc', 'zloc'])

# Set up a progress bar for iterating through the filtered annotations
pbar = tqdm(total=new_df.shape[0], position=1)

# Loop through each row in the filtered DataFrame and extract relevant information
for idx, row in new_df.iterrows():
    pbar.update(1)  # Update the progress bar
    # Check if the annotation file exists in the specified directory
    if os.path.exists(os.path.join("original_data/train_annots/", row['filename'])):
        # Populate result_df with bounding box coordinates, angle, and 3D location
        result_df.at[idx, 'filename'] = row['filename']
        result_df.at[idx, 'xmin'] = int(row['xmin'])
        result_df.at[idx, 'ymin'] = int(row['ymin'])
        result_df.at[idx, 'xmax'] = int(row['xmax'])
        result_df.at[idx, 'ymax'] = int(row['ymax'])
        result_df.at[idx, 'angle'] = row['observation angle']
        result_df.at[idx, 'xloc'] = int(row['xloc'])
        result_df.at[idx, 'yloc'] = int(row['yloc'])
        result_df.at[idx, 'zloc'] = int(row['zloc'])  # Distance (z-axis)

# Split the data into training and testing sets with a 90-10 split ratio
mask = np.random.rand(len(result_df)) < 0.9
train = result_df[mask]
test = result_df[~mask]

# Save the training and testing data into separate CSV files
train.to_csv('distance-estimator/data/train.csv', index=False)
test.to_csv('distance-estimator/data/test.csv', index=False)

print(f"\n\nData split completed. Training and testing CSV files are generated.")
