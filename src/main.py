#!/usr/bin/env python
from inference import *


if __name__ == "__main__":
    video_path = "src/videos/ice_hockey_processed.mp4"
    model_path = "yolov8s.pt"
    run_yolo_inference(video_path, model_path)
