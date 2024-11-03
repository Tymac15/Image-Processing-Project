def filter_highest_confidence_bounding_box(frames, bounding_boxes):
    filtered_bounding_boxes = []

    for frame_boxes in bounding_boxes:

        if frame_boxes:
            highest_conf_box = max(frame_boxes, key=lambda box: box["confidence"])
            filtered_bounding_boxes.append([highest_conf_box])
        else:
            filtered_bounding_boxes.append([])

    return frames, filtered_bounding_boxes
