
# Distance Estimation Using KITTI Dataset: Ground Truth Analysis

## Purpose
The primary goal of this experiment is to estimate the distance of objects (such as cars, pedestrians, trucks, etc.) from a camera based on the 2D bounding box coordinates detected in an image. The output is a distance measure (zloc) indicating the distance between the detected object and the camera. 

## Overview
We train a deep learning model that takes as input the 2D bounding box coordinates of detected objects and predicts the distance (zloc) to those objects in a scene. This task uses the KITTI Vision Benchmark Suite, which provides ground truth information about object locations in camera coordinates.

### Input
- Bounding box coordinates: (xmin, ymin, xmax, ymax)

### Output
- Estimated distance to the object: (zloc)

## Dataset Source Information
The dataset used for this experiment is the **KITTI Vision Benchmark Suite**, specifically the object detection subset. KITTI is a widely-used benchmark suite for autonomous driving tasks, including stereo vision, visual odometry, and 3D object detection. For this experiment, we focus on the 2D bounding boxes and corresponding ground truth data.

- **Dataset source**: [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Files Downloaded**:
  - **Left color images**: [Download (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
  - **Camera calibration files**: [Download (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
  - **Training labels**: [Download (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

We use a small subset of 10 example images for demonstration purposes.

## Libraries Used
- **Python**: Base scripting language
- **OpenCV**: Image processing and visualization (`cv2`)
- **Matplotlib**: Plotting and displaying images (`matplotlib.pyplot`)
- **Pandas**: Data handling and manipulation (`pandas`)
- **Tabulate**: Tabular data visualization (`tabulate`)
- **NumPy**: Numerical operations (`numpy`)
- **TQDM**: Progress visualization during processing (`tqdm`)
- **TensorFlow** and **Keras**: Deep learning libraries for training and model inference

## Scripts Description

### 1. **Data Preparation and Visualization**
- **`generate-csv.py`**: Converts the raw `.txt` label files into a structured CSV file (`annotations.csv`). It extracts essential information such as bounding box coordinates, object type, and 3D location data.
- **`generate-depth-annotations.py`**: Further processes the `annotations.csv` file to split it into `train.csv` and `test.csv` files. It filters out unnecessary data (e.g., "DontCare" objects) and organizes the dataset into a training and testing split for distance estimation.
- **`visualizer.py`**: Used to visualize bounding boxes on sample images from the dataset. This script overlays bounding boxes along with predicted distances to help debug and verify the data visually.

### 2. **Training the Model**
- **`hyperopti.py`**: Performs hyperparameter optimization using the Hyperas library. This step helps in selecting the best hyperparameters for training the model, such as learning rate, batch size, and optimizer. It supports single or multi-GPU configurations.
- **`train.py`**: Defines the deep learning model for distance estimation and starts the training process. You can choose your own model architecture or use hyperparameter-optimized parameters.

### 3. **Inference and Visualization**
- **`inference.py`**: Generates predictions on the test dataset using the trained model. The predictions (estimated distances) are stored in a CSV file for further analysis and visualization.
- **`prediction-visualizer.py`**: Visualizes the predictions by overlaying them on the original images. It also calculates and displays the percentage of predictions within a 10% tolerance of the ground truth distances.

## Ground Truth Information
The ground truth in this experiment comes from the **zloc** values provided in the KITTI dataset. These values are part of the KITTI annotation files and represent the z-coordinate of an object's location in the camera's 3D coordinate system. In simpler terms, **zloc** indicates how far the object is from the camera along the Z-axis. 

### Breakdown of Label Information:
Example line from KITTI label file (`000000.txt`):

```
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
```

**Breakdown**:
- **Object Class**: Pedestrian
- **Bounding Box Coordinates**: (xmin, ymin, xmax, ymax) = (712.40, 143.00, 810.73, 307.92)
- **3D Dimensions (height, width, length)**: (1.89, 0.48, 1.20)
- **3D Location (xloc, yloc, zloc)**: (1.84, 1.47, 8.41) → **zloc = 8.41** (Distance from the camera)
- **Rotation**: 0.01

### Important Points to Consider
- **Bounding Box Accuracy**: Ensure that the bounding boxes align accurately with the objects in the image. Any misalignment could affect distance estimation performance.
- **Ground Truth Accuracy**: The zloc values from the annotation files are the ground truth distances. Use these values to compare against model predictions and calculate error metrics.
- **Class Handling**: Multiple object classes such as "Car", "Truck", "Pedestrian", etc., should be visualized distinctly for better interpretation of results.

## Results and Interpretation
### Results
The results are displayed as images with bounding boxes drawn around detected objects. Each image also shows the ground truth and predicted distances (zloc values). 

### Interpretation of Accuracy
In our experiment, we achieved an accuracy of **59.17% within a 10% tolerance range** for the distance predictions. This means that around 59.17% of our model’s predictions were within 10% of the ground truth distances. While this accuracy might seem modest, it is an initial experiment and serves as a starting point for further refinement and improvement.

## Appendix
### Preparing Data
1. **Download the Dataset**: Follow the instructions in the dataset source section to download and organize the KITTI dataset.
2. **Organize Data**: Ensure that the data is organized as follows:
   ```
   KITTI-distance-estimation
   |-- original_data
       |-- test_images
       |-- train_annots
       `-- train_images
   ```
3. **Generate Annotations CSV**:
   ```shell
   python generate-csv.py --input=original_data/train_annots --output=annotations.csv
   ```
4. **Generate Train and Test CSV**:
   ```shell
   python generate-depth-annotations.py
   ```
5. **Visualize the Dataset**: Use `visualizer.py` to visualize and inspect the dataset for any issues.

### Running Training and Inference
1. **Hyperparameter Optimization**:
   ```shell
   python hyperopti.py
   ```
2. **Train the Model**:
   ```shell
   python train.py
   ```
3. **Run Inference**:
   ```shell
   python inference.py --modelname=generated_files/model.json --weights=generated_files/model.weights.h5
   ```
4. **Visualize Predictions**:
   ```shell
   python prediction-visualizer.py
   ```

## Acknowledgements
We extend our gratitude to the **KITTI Vision Benchmark Suite** for providing the dataset used in this experiment. For more information about the dataset, visit the [KITTI Dataset website](http://www.cvlibs.net/datasets/kitti/)

### Code Citation: <a href="https://github.com/harshilpatel312/KITTI-distance-estimation/tree/master" target="_blank">https://github.com/harshilpatel312/KITTI-distance-estimation/tree/master</a>

---

This experiment serves as an initial attempt at estimating object distances using 2D bounding box information. Further improvements and refinements can enhance the model’s accuracy and robustness. Feel free to explore, experiment, and contribute back to this project! 🚗📷
