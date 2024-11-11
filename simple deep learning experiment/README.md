
# Monocular Distance Estimation with KITTI Dataset: A Neural Network Approach

## Purpose
The purpose of this experiment is to estimate the distance (zloc) of objects (such as cars, pedestrians, trucks, etc.) from a camera using a neural network model trained on 2D bounding box coordinates. The model predicts the distance between detected objects and the camera based on 2D information extracted from images.

## Overview
In this project, we train a simple neural network model that takes 2D bounding box coordinates from detected objects in an image and predicts their distance from the camera (zloc). The experiment leverages the KITTI Vision Benchmark Suite, which provides ground truth 3D location information of objects, allowing us to evaluate the model's accuracy effectively.

### Project Structure
```
distance-estimator
├── data                    # Contains training and testing data
├── generated_files         # Stores generated model files
├── logs                    # Stores training logs for TensorBoard
├── output_images           # Output visualizations
├── original_data           # Raw KITTI dataset files
├── results                 # Directory for storing results
├── hyperopti.py            # Hyperparameter optimization script
├── inference.py            # Model inference script
├── plot_history.py         # Training history visualization script
├── train.py                # Main training script
├── training_continuer.py   # Script to continue interrupted training
├── generate-csv.py         # Converts KITTI labels to CSV format
├── generate-depth-annotations.py  # Splits data for depth estimation
├── visualizer.py           # Visualizes bounding boxes on images
├── prediction-visualizer.py # Visualizes model predictions
└── README.md               # Project documentation
```

### Input
- Bounding box coordinates: `(xmin, ymin, xmax, ymax)`

### Output
- Estimated distance to the object: `zloc`

## Dataset Source Information
The dataset used in this experiment is from the **KITTI Vision Benchmark Suite**, specifically the object detection subset. KITTI is a comprehensive benchmark for autonomous driving research. For this project, we use the 2D bounding boxes and corresponding ground truth data for distance estimation.

- **Dataset source**: [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
- **Required Files**:
  - **Left color images**: [Download (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
  - **Camera calibration files**: [Download (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
  - **Training labels**: [Download (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

We use a small subset of images for demonstration.

## Libraries and Dependencies
- **Python**: Programming language
- **OpenCV** (`cv2`): Image processing library
- **Matplotlib**: Data visualization library
- **Pandas**: Data manipulation and handling
- **NumPy**: Numerical operations
- **TQDM**: Progress visualization
- **TensorFlow** and **Keras**: Deep learning frameworks for model training
- **Hyperas**: Hyperparameter optimization

Install dependencies with:
```bash
pip install opencv-python matplotlib pandas numpy tqdm tensorflow keras hyperas
```

## Experiment Workflow

### Environment Setup
1. **Install Required Libraries** as mentioned above.
2. **Organize the Data Directory**:
   Download and organize your KITTI dataset as follows:
   ```
   distance-estimator
   ├── original_data
   │   ├── test_images
   │   ├── train_annots
   │   └── train_images
   ```

### Data Preparation and Visualization

1. **Convert Labels to CSV**:
   Use `generate-csv.py` to convert raw KITTI label files to a structured CSV file (`annotations.csv`) for easy data handling.
   ```bash
   python generate-csv.py --input=original_data/train_annots --output=annotations.csv
   ```

2. **Split Data for Training and Testing**:
   Run `generate-depth-annotations.py` to create `train.csv` and `test.csv`, filtering out irrelevant data (e.g., "DontCare" objects).
   ```bash
   python generate-depth-annotations.py
   ```

3. **Visualize Bounding Boxes**:
   Use `visualizer.py` to overlay bounding boxes on sample images to verify data accuracy.
   ```bash
   python visualizer.py
   ```

### Model Training and Hyperparameter Optimization

1. **Hyperparameter Tuning**:
   Run `hyperopti.py` to optimize hyperparameters such as learning rate, batch size, and optimizer. This step helps find the optimal configurations for training.
   ```bash
   python hyperopti.py
   ```

2. **Train the Model**:
   Use `train.py` to define and train the neural network model on the processed dataset.
   ```bash
   python train.py
   ```

3. **Continue Training (if interrupted)**:
   Use `training_continuer.py` to resume training from saved checkpoints.
   ```bash
   python training_continuer.py
   ```

### Model Inference and Visualization

1. **Run Predictions**:
   Use `inference.py` to generate distance predictions on the test dataset, storing the results in a CSV file.
   ```bash
   python inference.py --modelname=generated_files/model.json --weights=generated_files/model.weights.h5
   ```

2. **Visualize Predictions**:
   Run `prediction-visualizer.py` to overlay predictions on images, displaying both ground truth and estimated distances.
   ```bash
   python prediction-visualizer.py
   ```

### Ground Truth Information

Ground truth in this experiment is derived from **zloc** values in KITTI annotations, representing the z-coordinate of an object's location relative to the camera. This serves as the target variable during training.

Example line from KITTI label file (`000000.txt`):
```
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
```

**Interpretation**:
- **Object Class**: Pedestrian
- **Bounding Box Coordinates**: `(xmin, ymin, xmax, ymax) = (712.40, 143.00, 810.73, 307.92)`
- **3D Dimensions**: `(height, width, length) = (1.89, 0.48, 1.20)`
- **3D Location**: `(xloc, yloc, zloc) = (1.84, 1.47, 8.41)` → **zloc = 8.41** (distance from the camera)

### Important Considerations
- **Bounding Box Accuracy**: Misalignment in bounding boxes can affect the model’s accuracy in distance estimation.
- **Ground Truth Reliability**: Use `zloc` values for evaluating model performance.
- **Class-specific Analysis**: Visualize classes (e.g., Car, Pedestrian) distinctly to improve interpretability.

## Results and Interpretation

The results are visualized with bounding boxes drawn around detected objects, showing both the ground truth and predicted distances (`zloc` values).

**Accuracy**: We achieved **59.17% accuracy within a 10% tolerance range** for distance predictions, providing a solid foundation for further experimentation.

## Appendix

### Running the Full Experiment

1. **Prepare Dataset**:
   Download and structure data as instructed in the Dataset Source section.
2. **Generate Annotations**:
   ```bash
   python generate-csv.py --input=original_data/train_annots --output=annotations.csv
   ```
3. **Split Data for Training and Testing**:
   ```bash
   python generate-depth-annotations.py
   ```
4. **Train the Model**:
   ```bash
   python train.py
   ```
5. **Run Inference**:
   ```bash
   python inference.py --modelname=generated_files/model.json --weights=generated_files/model.weights.h5
   ```

##### Code Reference: [KITTI Distance Estimation GitHub](https://github.com/harshilpatel312/KITTI-distance-estimation/tree/master)

## Acknowledgements

We extend our gratitude to the **KITTI Vision Benchmark Suite** for the dataset. More information is available at the [KITTI Dataset website](http://www.cvlibs.net/datasets/kitti/).

---

This experiment is an introductory attempt at estimating object distances using 2D bounding box coordinates and simple neural network models. Further improvements could significantly enhance accuracy and robustness.
