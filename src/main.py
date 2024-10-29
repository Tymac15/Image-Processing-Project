#!/usr/bin/env python
from inference import *
from thresholding import *

if __name__ == "__main__":
    video_path = "src/videos/ice_hockey_processed.mp4"
    model_path = "src/models/ice_hockey_model.pt"
    
    frames, bounding_boxes = run_yolo_inference(video_path, model_path)

    frames, filtered_bounding_boxes = filter_highest_confidence_bounding_box