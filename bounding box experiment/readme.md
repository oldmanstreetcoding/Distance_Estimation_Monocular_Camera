# KITTI Dataset Bounding Box Experiment

2d_bound_boxes.ipynb is the  file for this experiment

## Project Overview
This repository is an exploration of the KITTI 3D object detection dataset, focusing on the extraction and visualization of 2D bounding boxes from the provided images and labels. The KITTI dataset is a widely-used benchmark suite for autonomous driving tasks, including stereo, optical flow, visual odometry, 3D object detection, and 3D tracking. The main goal of this experiment is to display and overlay bounding boxes on sample images from the dataset using Python.

## Dataset Information
KITTI Dataset Source: The dataset used in this repository is part of the KITTI Vision Benchmark Suite, which was developed by Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. The dataset consists of left and right camera images, calibration files, and labeled ground truth data. You can download the dataset from the following links:

- Left Color Images (12 GB): <a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip" target="_blank">Download here</a>
- Right Color Images (12 GB): <a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip" target="_blank">Download here (Optional for stereo use)</a>
- Camera Calibration Files (16 MB): <a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip" target="_blank">Download here</a>
- Training Labels (5 MB): <a href="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip" target="_blank">Download here</a>

In this repository, we are using a small subset of the dataset for demonstration purposes, focusing on 10 sample images.

## Libraries Used
The following libraries and tools were used in this project:

- Python: Base language for scripting
- OpenCV: For image processing and visualization (cv2)
- Matplotlib: For plotting images (matplotlib.pyplot)
- Pandas: For data handling and manipulation (pandas)
- Tabulate: For tabular visualization of data (tabulate)
- NumPy: For numerical operations (numpy)
- TQDM: For progress visualization during processing (tqdm)

## Scripts Description

**2D_bounding_boxes.ipynb**

This Jupyter Notebook script explores the KITTI 3D detection dataset. It displays images, visualizes bounding boxes, and overlays information extracted from the labels. Here’s what the script accomplishes:

- Reading image files and labels: The script uses the provided directories to load images and their corresponding labels.
- Displaying images: Loaded images are displayed using Matplotlib for verification.
- Reading and parsing labels: The script reads the 2D bounding box labels from the label files. It extracts relevant fields such as object class, bounding box coordinates (xmin, ymin, xmax, ymax), and annotation attributes.
- Overlaying bounding boxes: It draws the bounding boxes on the corresponding images using the extracted label information. Additionally, the image is overlaid with information such as the bounding box ID.
- Stereo camera image processing (Optional): If stereo images are provided, the script combines left and right images using weighted overlays to visualize stereo effects.

## Key Considerations and Challenges
- Precision of Bounding Box Coordinates: Ensuring that the bounding boxes align precisely with the objects in the images is crucial for accuracy.
- Handling Multiple Object Classes: The dataset contains multiple classes of objects with varying sizes. Visualizations need to consider the scale and type of each object to avoid clutter.
- Camera Calibration: If working with stereo or 3D data, ensuring that camera calibration parameters are correctly applied is essential for accurate measurements.

## Additional Information
This project serves as an introductory experiment to visualize and process bounding boxes on the KITTI dataset using Python and common computer vision libraries. Further work can include:

- Advanced Object Detection: Using pre-trained models like YOLO or Faster R-CNN to detect objects.
- Depth Estimation: Working with stereo images to estimate depth information.
- Performance Metrics: Measuring detection accuracy using metrics like Intersection over Union (IoU).

Feel free to explore the provided scripts, modify them to suit your needs, and contribute back to this repository with your findings or improvements!

## Acknowledgments
This project uses the KITTI Vision Benchmark Suite dataset. We thank the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago for providing the dataset. More information about the dataset can be found on the KITTI Dataset website.

Happy experimenting with KITTI! 🚗