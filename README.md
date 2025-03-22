Advanced Driver Assistance Systems (ADAS) - Multi-Sensor Fusion

Overview

This project focuses on developing an Advanced Driver Assistance System (ADAS) using multi-sensor fusion techniques. The system integrates camera, LiDAR, and radar data to perform object detection, lane detection, and collision avoidance, enhancing autonomous vehicle safety and decision-making.

Features

Multi-Sensor Fusion: Integrates data from camera, LiDAR, and radar for improved perception.

Object Detection: Identifies vehicles, pedestrians, and obstacles using deep learning models.

Lane Detection: Detects road lanes for lane-keeping assistance.

Collision Avoidance: Issues warnings and takes preventive measures to avoid collisions.

Adaptive Cruise Control: Adjusts vehicle speed based on detected objects and distances.

Dataset

This project uses the nuScenes mini dataset, which provides real-world sensor data for autonomous driving.

Download Dataset: nuScenes mini dataset

Dataset Includes:

Camera images

LiDAR point clouds

Radar data

Annotations for object detection

Installation & Requirements

Python Version

This project requires Python 3.8 or higher.

Install Dependencies

To install all required dependencies, run:

pip install -r requirements.txt

Running the Project

Clone the repository:

git clone https://github.com/your-username/ADAS-Multi-Sensor-Fusion.git

Navigate to the project directory:

cd ADAS-Multi-Sensor-Fusion

Run the main script:

python main.py

Results & Visualizations

Lane detection results are displayed with lane markings on road images.

Object detection results highlight detected objects with bounding boxes.

LiDAR data is visualized in 3D for better spatial understanding.

Collision warnings and alerts are triggered based on real-time sensor data.

License

This project is open-source and available under the MIT License.
