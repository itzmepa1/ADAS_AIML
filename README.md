
# Advanced Driver Assistance Systems (ADAS) - Multi-Sensor Fusion

## Overview

This project focuses on developing an Advanced Driver Assistance System (ADAS) using multi-sensor fusion techniques. The system integrates camera, LiDAR, and radar data to perform object detection, lane detection, and collision avoidance, enhancing autonomous vehicle safety and decision-making.

## Features

- **Multi-Sensor Fusion:** Integrates data from camera, LiDAR, and radar for improved perception.
- **Object Detection:** Identifies vehicles, pedestrians, and obstacles using deep learning models.
- **Lane Detection:** Detects road lanes for lane-keeping assistance.
- **Collision Avoidance:** Issues warnings and takes preventive measures to avoid collisions.
- **Adaptive Cruise Control:** Adjusts vehicle speed based on detected objects and distances.

## Dataset

This project uses the **nuScenes mini dataset**, which provides real-world sensor data for autonomous driving and also the dataset should be downloaded and placed inside the project folder to ensure the code runs smoothly.

- **Download Dataset:** [nuScenes mini dataset](https://www.nuscenes.org/download)
- **Dataset Includes:**
  - Camera images
  - LiDAR point clouds
  - Radar data
  - Annotations for object detection

## Installation & Requirements

### Python Version

- This project requires **Python 3.8 or higher**.

### Install Dependencies

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/itzmepa1/ADAS_AIML.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ADAS_AIML
   ```
3. Run the main script:
   ```bash
   python adas_aiml.py
   ```

## Results & Visualizations
- Lane detection results are displayed with lane markings on road images.
- Object detection results highlight detected objects with bounding boxes.
- LiDAR data is visualized in 3D for better spatial understanding.
- Collision warnings and alerts are triggered based on real-time sensor data.

## **Improvements to Increase Efficiency **

### **1. Sensor Fusion Optimization**
**Current Issue:**  
Individual sensors (LiDAR, Radar, Camera) provide separate data, which may introduce latency or inconsistencies.

**Improvement:**  
🔹 Implement **Kalman Filtering** or **Bayesian Filtering** to fuse sensor data more effectively.  
🔹 This will help reduce noise, enhance accuracy, and improve real-time object detection.

---

### **2. Model Acceleration & Optimization**
**Current Issue:**  
YOLOv8 may introduce latency in real-time applications due to high computational requirements.

**Improvement:**  
🔹 Utilize **TensorRT** or **ONNX Runtime** for deep learning model optimization.  
🔹 This will accelerate inference speed on GPUs, significantly reducing prediction time.

---

### **3. Adaptive Data Sampling & Processing**
**Current Issue:**  
Processing full-frame LiDAR point clouds or high-resolution images slows down prediction speed.

**Improvement:**  
🔹 **Voxelization** can be used to reduce LiDAR point cloud size while preserving key features.  
🔹 **Region of Interest (ROI) Filtering** can limit processing to relevant areas in camera images, avoiding unnecessary computations.

These enhancements will **improve efficiency, reduce latency, and maintain high accuracy**, making the system more reliable for real-time ADAS applications. 



---



