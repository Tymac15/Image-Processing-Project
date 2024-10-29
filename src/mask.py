import cv2
import numpy as np

def apply_bounding_box_mask(
    frames, 
    bounding_boxes, 
    background_color=(0, 0, 0), 
    background_alpha=0.2, 
    oval_color=(255, 255, 255), 
    oval_alpha=0.3
):
    """
    Applies a darkened background and highlights the bounding box area with a transparent oval.
    
    Parameters:
    - frames: List of frames to process.
    - bounding_boxes: List of bounding box lists per frame.
    - background_color: Tuple for the background darkening color (default black).
    - background_alpha: Alpha transparency for the background darkening (default 0.8).
    - oval_color: Tuple for the oval highlight color (default white).
    - oval_alpha: Alpha transparency for the oval highlight (default 0.7).
    """
    masked_frames = []

    for frame, boxes in zip(frames, bounding_boxes):
        # Darken the entire frame using the specified background color and alpha
        background = np.full_like(frame, background_color, dtype=np.uint8)
        darkened_frame = cv2.addWeighted(frame, 1 - background_alpha, background, background_alpha, 0)
        
        if boxes:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box["box"])

            # Calculate the center and axes lengths for the oval
            oval_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            oval_axes = ((x2 - x1) // 2, (y2 - y1) // 2)

            # Create an overlay with a transparent oval in the bounding box area
            overlay = darkened_frame.copy()
            cv2.ellipse(
                overlay,
                oval_center,          # Center of the oval
                oval_axes,            # Axes lengths (half-width and half-height)
                angle=0,              # No rotation
                startAngle=0,         # Start angle of the oval
                endAngle=360,         # End angle of the oval (full circle)
                color=oval_color,     # Oval color
                thickness=-1          # Filled oval
            )
            
            # Blend the overlay with the darkened frame to highlight the bounding box area
            highlighted_frame = cv2.addWeighted(overlay, oval_alpha, darkened_frame, 1 - oval_alpha, 0)
        else:
            # If no bounding box, keep the entire frame darkened
            highlighted_frame = darkened_frame

        # Append the processed frame
        masked_frames.append(highlighted_frame)

    return masked_frames
