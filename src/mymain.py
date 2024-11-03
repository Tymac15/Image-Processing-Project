#!/usr/bin/env python
from utils import *
from inference import *
from thresholding import *
from interpolate import *
from mask import *


if __name__ == "__main__":

    task = "field_hockey"

    

    video_path = f"C:/Users/tydav/Desktop/IP project/Image-Processing-Project/src/videos/{task}_processed.mp4"
    model_path = f"C:/Users/tydav/Desktop/IP project/Image-Processing-Project/src/models/{task}_model.pt"

    frames, bounding_boxes = run_yolo_inference(video_path, model_path, conf=0.4)


    frames, filtered_bounding_boxes = filter_highest_confidence_bounding_box(
        frames, bounding_boxes
    )

    frames, interpolated_bounding_boxes = interpolate_bounding_boxes(
        frames, bounding_boxes, max_gap=5
    )

    frames = masked_frames = apply_bounding_box_mask(
        frames,
        interpolated_bounding_boxes,
        background_color=(0, 0, 0),
        background_alpha=0.4,
        oval_color=(255, 255, 255),
        oval_alpha=0.3,
    )

    export_frames_to_video(
        frames, output_path=f"C:/Users/tydav/Desktop/IP project/Image-Processing-Project/src/output/{task}_highlighted.mp4", fps=25
    )
