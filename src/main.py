#! /usr/bin/python3
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from models.yolo import Model  # Import YOLOv5 model architecture
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)  # Import YOLO utilities
from utils.loss import ComputeLoss  # YOLOv5 loss function
from utils.torch_utils import select_device, time_sync  # Utility functions for training
import cv2
from glob import glob

from data_processor import *

data_dir = "data/"  # Change this to the path of your custom dataset
train_path = os.path.join(data_dir, "train_val/train")
val_path = os.path.join(data_dir, "train_val/validation")

# Set up device and model
device = select_device("GPU")  # Use GPU if available
model = torch.load("yolov5s.pt", map_location=device)  # Load pre-trained YOLOv5s model
num_classes = 2  # Change this to match the number of classes in your dataset

# Modify the YOLO model's output layer to match the number of custom classes
model.model[-1].nc = num_classes  # Update number of classes in model
model.model[-1].no = num_classes * (
    5 + num_classes
)  # Update the number of outputs (5 box attributes + classes)
model.names = ["hockey_puck", "hockey_ball"]  # Update class names

# Create a custom loss function
loss_fn = ComputeLoss(model)  # YOLOv5 custom loss function

train_image_dir = os.path.join(train_path, "images")
train_labels_dir = os.path.join(train_path, "labels")
validation_image_dir = os.path.join(train_path, "images")
validation_labels_dir = os.path.join(train_path, "labels")

# Set up the DataLoader
train_dataset = YOLODataset(image_dir=train_image_dir, label_dir=train_labels_dir)
val_dataset = YOLODataset(
    image_dir=validation_image_dir, label_dir=validation_labels_dir
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=-1)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=-1)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
