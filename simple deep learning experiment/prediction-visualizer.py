'''
Purpose: visualize data from the dataframe
'''
import os
import cv2
import pandas as pd

# Load the prediction results
df = pd.read_csv("distance-estimator/data/predictions.csv")

for idx, row in df.iterrows():
    if os.path.exists(os.path.join("original_data/train_annots/", row['filename'])):
        # Read the image
        fp = os.path.join("original_data/train_images", row['filename'].replace('.txt', '.png'))
        im = cv2.imread(fp)

        # Get bounding box coordinates
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        # Draw a center line (optional visual aid)
        cv2.line(im, (int(1224/2), 0), (int(1224/2), 370), (255, 255, 255), 2)
        # Draw the bounding box around the detected object
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate the percentage error between actual and predicted zloc
        actual = row['zloc']
        predicted = row['zloc_pred']
        percentage_error = abs(actual - predicted) / actual * 100 if actual != 0 else 0

        # Prepare the text to display
        info_string = "(act: {:.2f}, pred: {:.2f}, err: {:.2f}%)".format(actual, predicted, percentage_error)

        # Display the information at the center of the bounding box
        cv2.putText(im, info_string, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the image with the bounding box and additional information
        cv2.imshow("detections", im)
        cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()