import os
import numpy as np
import cv2
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.utils.splits import create_splits_scenes
import pygame  # Using pygame for audio alerts

# Initialize pygame mixer
pygame.mixer.init()
alert_sound = "alert.wav"  

# Load nuScenes dataset (Set dataset path)
nusc = NuScenes(version='v1.0-mini', dataroot=r'C:\Users\pavan\OneDrive\Desktop\ADAS_AIML-Assignment\nuscenes_dataset', verbose=True)

# Load a sample scene
# Change the scene index to load a different scene from 0 to 9
# The nuScenes dataset consists of multiple scenes, each containing multiple samples.

index_scene=int(input("Enter the index scene from 0 to 9:"))
scene = nusc.scene[index_scene]  
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

# Load Camera, Radar, and LiDAR data

# Change the camera token to use different camera views as required.
# Available options (based on dataset structure) include:
# 'CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
camera_token = sample['data']['CAM_FRONT']

# Change the radar token to use different radar views as required.
# Available options (based on dataset structure) include:
# 'RADAR_FRONT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT'
radar_token = sample['data']['RADAR_FRONT']

# The LiDAR token should not be changed as there is only one LiDAR sensor available in the dataset.
# The available option is:
# 'LIDAR_TOP'
lidar_token = sample['data']['LIDAR_TOP']

cam_data = nusc.get('sample_data', camera_token)
radar_data = nusc.get('sample_data', radar_token)
lidar_data = nusc.get('sample_data', lidar_token)

# Load and display camera image
img_path = os.path.join(nusc.dataroot, cam_data['filename'])
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title("Camera View")
plt.show()

# Load Radar Data
radar_path = os.path.join(nusc.dataroot, radar_data['filename'])
radar_point_cloud = RadarPointCloud.from_file(radar_path)
radar_points = radar_point_cloud.points[:3, :].T  # Extract x, y, z coordinates

# Load LiDAR Data
lidar_path = os.path.join(nusc.dataroot, lidar_data['filename'])
lidar_point_cloud = LidarPointCloud.from_file(lidar_path)
lidar_points = lidar_point_cloud.points[:3, :].T  # Extract x, y, z coordinates

# Display LiDAR Point Cloud
o3d_pcd = o3d.geometry.PointCloud()
o3d_pcd.points = o3d.utility.Vector3dVector(lidar_points)
o3d.visualization.draw_geometries([o3d_pcd], window_name="LiDAR Point Cloud")

# Load YOLOv8 Model for Real-Time Object Detection
yolo_model = YOLO("yolov8n.pt")
results = yolo_model.predict(img_path, save=False)

# Draw Bounding Boxes
def draw_yolo_boxes(image, results):
    img = np.array(image)
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

image_with_boxes = draw_yolo_boxes(image, results)
plt.imshow(image_with_boxes)
plt.title("YOLOv8 Object Detection")
plt.show()

# Decision Making: Collision Avoidance
def collision_avoidance(radar_points, lidar_points, threshold_distance=5.0):
    for point in np.vstack((radar_points, lidar_points)):
        x, y = point[:2]
        distance = np.sqrt(x**2 + y**2)
        if distance < threshold_distance:
            pygame.mixer.Sound(alert_sound).play()
            return "ALERT: Collision Risk! Applying Brakes!"
    return "No Collision Risk."

decision = collision_avoidance(radar_points, lidar_points)
print(decision)

# Lane Detection using OpenCV
def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image

lane_image = detect_lanes(image_with_boxes)
plt.imshow(lane_image)
plt.title("Lane Detection")
plt.show()

# Adaptive Cruise Control (ACC) Simulation
def adaptive_cruise_control(radar_points, safe_distance=10.0):
    for point in radar_points:
        distance = np.sqrt(point[0]**2 + point[1]**2)
        if distance < safe_distance:
            return "Slowing Down: Adaptive Cruise Control Activated!"
    return "Maintaining Speed."

acc_decision = adaptive_cruise_control(radar_points)
print(acc_decision)

# Lane Change Assistance
def lane_change_assistance(lane_image, radar_points, min_safe_gap=5.0):
    left_clear = right_clear = True
    for point in radar_points:
        x, y = point[:2]
        if -min_safe_gap < x < 0:  # Object on left
            left_clear = False
        if 0 < x < min_safe_gap:  # Object on right
            right_clear = False
    
    if left_clear:
        return "Safe to Change Lane to Left."
    elif right_clear:
        return "Safe to Change Lane to Right."
    else:
        return "No Safe Lane Change Possible. Maintain Lane."

lane_change_decision = lane_change_assistance(lane_image, radar_points)
print(lane_change_decision)
