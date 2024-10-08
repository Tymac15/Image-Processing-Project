import torch
import os
import cv2
from glob import glob


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