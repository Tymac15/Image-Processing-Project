#!/usr/bin/env python
from utils import *
from inference import *
from thresholding import *
from interpolate import *
from mask import *

import cv2


def export_video_with_bounding_boxes(
    frames, bounding_boxes, output_path="output_with_boxes.mp4", fps=25
):
    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate over frames and bounding boxes
    for frame, boxes in zip(frames, bounding_boxes):
        frame_copy = frame.copy()  # Copy frame to avoid modifying the original

        # Draw bounding box if present
        if boxes:
            box = boxes[0]  # Only one box per frame after filtering
            x1, y1, x2, y2 = map(int, box["box"])

            # Draw rectangle around the bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

        # Write the frame with bounding box to the video
        out.write(frame_copy)

    # Release the VideoWriter object
    out.release()
    print(f"Video with bounding boxes saved to {output_path}")


if __name__ == "__main__":
    video_path = "src/videos/ice_hockey_processed.mp4"
    model_path = "src/models/ice_hockey_model.pt"

    frames, bounding_boxes = run_yolo_inference(video_path, model_path)

    export_video_with_bounding_boxes(
        frames, bounding_boxes, output_path="video_with_boxes.mp4", fps=25
    )

    frames, filtered_bounding_boxes = filter_highest_confidence_bounding_box(
        frames, bounding_boxes
    )

    export_video_with_bounding_boxes(
        frames,
        filtered_bounding_boxes,
        output_path="filtered_video_with_boxes.mp4",
        fps=25,
    )

    frames, interpolated_bounding_boxes = interpolate_bounding_boxes(
        frames, bounding_boxes, max_gap=10
    )

    # export_video_with_bounding_boxes(
    #     frames,
    #     interpolated_bounding_boxes,
    #     output_path="interpolated_video_with_boxes.mp4",
    #     fps=25,
    # )

    frames = masked_frames = apply_bounding_box_mask(
        frames,
        bounding_boxes,
        background_color=(0, 0, 0),
        background_alpha=0.4,
        oval_color=(255, 255, 255),
        oval_alpha=0.3,
    )
    
    export_frames_to_video(
        frames, output_path="highlighted_video_with_boxes.mp4", fps=25
    )
    # export_frames_to_video(frames, output_path='src/output/highlighted_video.mp4', fps=25)
