def interpolate_bounding_boxes(frames, bounding_boxes, max_gap=5):
    
    
    # Initialize interpolated bounding boxes
    interpolated_bounding_boxes = bounding_boxes[:]

    # Iterate over each frame
    i = 0
    while i < len(bounding_boxes):
        if not bounding_boxes[i]:  # If no bounding box in the current frame
            # Find the next frame with a bounding box within the max_gap
            start = i - 1  # Last known frame with a bounding box
            # Check if `start` has a bounding box
            if start < 0 or not bounding_boxes[start]:
                i += 1
                continue
            
            end = i
            while end < len(bounding_boxes) and not bounding_boxes[end] and end - start <= max_gap:
                end += 1
            
            # Ensure `end` is within range and contains a bounding box
            if end < len(bounding_boxes) and bounding_boxes[end] and end - start <= max_gap:
                # Get starting and ending bounding box coordinates for interpolation
                start_box = bounding_boxes[start][0]
                end_box = bounding_boxes[end][0]
                
                # Linearly interpolate each missing frame
                for j in range(1, end - start):
                    interp_box = {
                        'class_id': start_box['class_id'],  # Assuming class remains the same
                        'confidence': min(start_box['confidence'], end_box['confidence']),  # Take lower confidence
                        'box': (
                            start_box['box'][0] + (end_box['box'][0] - start_box['box'][0]) * j / (end - start),
                            start_box['box'][1] + (end_box['box'][1] - start_box['box'][1]) * j / (end - start),
                            start_box['box'][2] + (end_box['box'][2] - start_box['box'][2]) * j / (end - start),
                            start_box['box'][3] + (end_box['box'][3] - start_box['box'][3]) * j / (end - start)
                        )
                    }
                    interpolated_bounding_boxes[start + j] = [interp_box]
                
            # Move to the end of the interpolated section
            i = end
        else:
            # If bounding box exists, just move to the next frame
            i += 1

    return frames, interpolated_bounding_boxes
