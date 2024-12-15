'''
Purpose: Visualize data from the dataframe.
This script loads image and bounding box data from a CSV file, draws bounding boxes around objects in the images, 
and overlays depth estimation information such as angle and distance (zloc) for each detected object.
'''

import os
import cv2
import pandas as pd

# Load the training data CSV file containing bounding box and object information
df = pd.read_csv("distance-estimator/data/train.csv")

# Iterate over each row in the DataFrame to access image file names and bounding box information
for idx, row in df.iterrows():
    if os.path.exists(os.path.join("original_data/train_images/", row['filename'])):
        fp = os.path.join("original_data/train_images/", row['filename'].replace('.txt', '.png'))
        im = cv2.imread(fp)

        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        cv2.line(im, (int(1224/2), 0), (int(1224/2), 370), (255,255,255), 2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        string = "({}, {})".format(row['angle'], row['zloc'])
        cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Save the output image with bounding boxes and labels
        output_fp = os.path.join("output_images", f"processed_{row['filename']}")
        cv2.imwrite(output_fp, im)
        print(f"Saved processed image to {output_fp}")
