import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
from models.yolo import Model  # Import YOLOv5 model architecture
from utils.general import check_img_size, non_max_suppression, scale_coords  # Import YOLO utilities
from utils.loss import ComputeLoss  # YOLOv5 loss function
from utils.torch_utils import select_device, time_sync  # Utility functions for training
import cv2
from glob import glob

# Define paths
data_dir = 'dataset/'  # Change this to the path of your custom dataset
train_image_path = os.path.join(data_dir, 'images/train')
val_image_path = os.path.join(data_dir, 'images/val')

# Create a function to load the YOLO dataset
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_files = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.label_files = sorted(glob(os.path.join(label_dir, '*.txt')))
        self.img_size = img_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Load labels
        label_path = self.label_files[idx]
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert YOLO format to bounding box (x_min, y_min, x_max, y_max)
                    x_center, y_center, width, height = x_center * w, y_center * h, width * w, height * h
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    boxes.append([class_id, x_min, y_min, x_max, y_max])

        # Convert image to tensor and resize
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

        # Convert boxes to tensor
        labels = torch.tensor(boxes, dtype=torch.float32)
        return img, labels

# Set up device and model
device = select_device('')  # Use GPU if available
model = torch.load('yolov5s.pt', map_location=device)  # Load pre-trained YOLOv5s model
num_classes = 2  # Change this to match the number of classes in your dataset

# Modify the YOLO model's output layer to match the number of custom classes
model.model[-1].nc = num_classes  # Update number of classes in model
model.model[-1].no = num_classes * (5 + num_classes)  # Update the number of outputs (5 box attributes + classes)
model.names = ['hockey_puck', 'hockey_ball']  # Update class names

# Create a custom loss function
loss_fn = ComputeLoss(model)  # YOLOv5 custom loss function

# Set up the DataLoader
train_dataset = YOLODataset(image_dir=train_image_path, label_dir=os.path.join(data_dir, 'labels/train'))
val_dataset = YOLODataset(image_dir=val_image_path, label_dir=os.path.join(data_dir, 'labels/val'))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        pred = model(images)

        # Compute loss
        loss, loss_items = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f}')

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_yolov5s.pth')
