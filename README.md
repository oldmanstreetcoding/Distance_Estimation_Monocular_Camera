# Monoviz: Distance Estimation Using Geometry-Based Distance Decomposition for Monocular 3D Object Detection

## Overview

This repository is part of a capstone project aimed at developing a **cost-effective distance estimation solution** using a **monocular camera**. The project is motivated by the need to provide affordable **Advanced Driver Assistance Systems (ADAS)** for achieving **Level 3+ Automation**, thereby reducing the dependency on expensive sensors like LiDAR and RADAR. 

The proposed solution integrates **analytical** and **deep learning approaches** to achieve real-time distance estimation for multiple objects in a scene, targeting both longitudinal and lateral distances.

---

## Project Objectives

### Key Goals:
1. **Problem Statement**: Estimate the distance between the ego vehicle and target objects using a monocular camera.
2. **Output**: Graphical representation, including **bird's-eye view** visualizations of estimated distances.
3. **System Capabilities**:
   - Handle multiple objects in real-time.
   - Predict both **longitudinal (X)** and **lateral (Y)** distances.
4. **Approach**:
   - Combine **analytical** and **deep learning** methods for efficiency and accuracy.
   - Ensure modularity for integration into existing ADAS systems.

### Why Monocular Cameras?
- Cost-effective compared to LiDAR and RADAR:
  - **LiDAR Cost**: $1,000–$10,000 per unit.
  - **Monocular Camera Cost**: $100–$300 per unit.
- Environmentally sustainable with a smaller ecological footprint.

### Performance Metrics:
1. **Accuracy within 5% Margin** for distance prediction.
2. **Mean Absolute Error (MAE)**:
   - Overall MAE.
   - MAE for objects closer or farther than 30 meters.

---

## Repository Structure

This repository contains **three key sub-projects**, each contributing to the overall solution:

1. **Bounding Box Experiment**:
   - Focused on simple bounding box-based distance estimation.
   - Explores the limitations of relying solely on bounding box parameters and object class for distance prediction.

2. **Simple Neural Network Experiment**:
   - Investigates the use of a lightweight neural network for monocular distance estimation.
   - Incorporates distance prediction as a regression task, leveraging bounding box coordinates and object class as inputs.

3. **Final Solution with MonoViz**:
   - Combines advanced visualization techniques (e.g., bird's-eye view) with refined deep learning approaches.
   - The most robust and accurate implementation in this repository.

---

## Future Work

1. **Enhanced Feature Extraction**:
   - Incorporate monocular depth cues and semantic segmentation.
   - Explore transformer-based architectures for improved context understanding.
2. **Expanded Dataset**:
   - Use diverse datasets to handle varied lighting and weather conditions.
3. **Real-World Integration**:
   - Test the algorithm in real-world ADAS scenarios with embedded hardware.

---

## Acknowledgements

This work builds upon [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) and related open-source repositories, including [KITTI-Distance-Estimation](https://github.com/harshilpatel312/KITTI-distance-estimation).

---

## Contributions

We welcome contributions to enhance the project. Feel free to submit issues or pull requests.

---

This README serves as a comprehensive guide to the repository, detailing our capstone project, its objectives, implementation details, and future directions.
