import cv2
import numpy as np


def apply_bounding_box_mask(
    frames,
    bounding_boxes,
    background_color=(0, 0, 0),
    background_alpha=0.2,
    oval_color=(255, 255, 255),
    oval_alpha=0.3,
):
    masked_frames = []

    for frame, boxes in zip(frames, bounding_boxes):

        background = np.full_like(frame, background_color, dtype=np.uint8)
        darkened_frame = cv2.addWeighted(
            frame, 1 - background_alpha, background, background_alpha, 0
        )

        if boxes:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box["box"])

            oval_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            oval_axes = ((x2 - x1) // 2, (y2 - y1) // 2)

            overlay = darkened_frame.copy()
            cv2.ellipse(
                overlay,
                oval_center,
                oval_axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=oval_color,
                thickness=-1,
            )

            highlighted_frame = cv2.addWeighted(
                overlay, oval_alpha, darkened_frame, 1 - oval_alpha, 0
            )
        else:

            highlighted_frame = darkened_frame

        masked_frames.append(highlighted_frame)

    return masked_frames
