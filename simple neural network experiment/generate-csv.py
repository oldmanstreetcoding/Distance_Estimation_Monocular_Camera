'''
Purpose: Generates a CSV file of annotations from .txt files in the specified input directory.
This script reads annotation files from the KITTI dataset, extracts relevant information 
such as bounding box coordinates, object class, dimensions, and 3D location, and saves them 
in a structured CSV file for easier data processing and analysis.
'''

import pandas as pd
import os
from tqdm import tqdm
import argparse

# Set up command-line arguments for input and output paths
argparser = argparse.ArgumentParser(description='Generate annotations CSV file from .txt files')
argparser.add_argument('-i', '--input', help='input directory name')
argparser.add_argument('-o', '--output', help='output file name')

args = argparser.parse_args()

# Parse arguments and define input/output paths
INPUTDIR = args.input
FILENAME = args.output

'''
KITTI Annotation Format:
Values    Name      Description
---------------------------------------------------------------------------
   1    type         Object type (e.g., 'Car', 'Van', 'Truck', 'Pedestrian')
   1    truncated    Float [0, 1] representing how much of the object is truncated by image boundaries
   1    occluded     Occlusion state (0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown)
   1    alpha        Observation angle of object [-pi, pi]
   4    bbox         Bounding box in 2D image (xmin, ymin, xmax, ymax)
   3    dimensions   3D object dimensions (height, width, length) in meters
   3    location     3D object location (x, y, z) in camera coordinates in meters
   1    rotation_y   Rotation around Y-axis in camera coordinates [-pi, pi]
   1    score        Detection confidence (only for detection results, not ground truth)
'''

# Initialize a DataFrame to store the parsed data with appropriate columns
df = pd.DataFrame(columns=['filename', 'class', 'truncated', 'occluded', 'observation angle', 
                           'xmin', 'ymin', 'xmax', 'ymax', 'height', 'width', 'length', 
                           'xloc', 'yloc', 'zloc', 'rot_y'])

def assign_values(filename, idx, list_to_assign):
    """
    Helper function to assign values from a parsed annotation line to the DataFrame.
    
    Parameters:
    - filename: The name of the annotation file being processed.
    - idx: The current row index in the DataFrame.
    - list_to_assign: List of values extracted from the annotation line.
    """
    # Assign basic information and bounding box coordinates
    df.at[idx, 'filename'] = filename
    df.at[idx, 'class'] = list_to_assign[0]
    df.at[idx, 'truncated'] = list_to_assign[1]
    df.at[idx, 'occluded'] = list_to_assign[2]
    df.at[idx, 'observation angle'] = list_to_assign[3]

    # Bounding box coordinates
    df.at[idx, 'xmin'] = list_to_assign[4]
    df.at[idx, 'ymin'] = list_to_assign[5]
    df.at[idx, 'xmax'] = list_to_assign[6]
    df.at[idx, 'ymax'] = list_to_assign[7]

    # 3D dimensions of the object (height, width, length)
    df.at[idx, 'height'] = list_to_assign[8]
    df.at[idx, 'width'] = list_to_assign[9]
    df.at[idx, 'length'] = list_to_assign[10]

    # 3D location of the object (x, y, z coordinates in camera coordinates)
    df.at[idx, 'xloc'] = list_to_assign[11]
    df.at[idx, 'yloc'] = list_to_assign[12]
    df.at[idx, 'zloc'] = list_to_assign[13]

    # Rotation angle around Y-axis in camera coordinates
    df.at[idx, 'rot_y'] = list_to_assign[14]

# List all files in the input directory and initialize a progress bar
all_files = sorted(os.listdir(INPUTDIR))
pbar = tqdm(total=len(all_files), position=1)

count = 0  # Initialize a counter for rows in the DataFrame

# Iterate over each file in the directory
for idx, f in enumerate(all_files):
    pbar.update(1)  # Update progress bar
    file_object = open(INPUTDIR + f, 'r')
    file_content = [x.strip() for x in file_object.readlines()]

    # Process each line in the annotation file
    for line in file_content:
        elements = line.split()
        
        # Skip lines with 'DontCare' objects, as these are not included in the analysis
        if elements[0] == 'DontCare':
            continue

        # Assign parsed values to the DataFrame for valid object annotations
        assign_values(f, count, elements)
        count += 1  # Increment the row index

# Save the completed DataFrame to a CSV file
df.to_csv(FILENAME, index=False)
print(f"\nAnnotations successfully saved to {FILENAME}")
