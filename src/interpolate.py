def interpolate_bounding_boxes(frames, bounding_boxes, max_gap=5):
    if max_gap <= 0:
        return frames, bounding_boxes

    interpolated_bounding_boxes = bounding_boxes[:]
    n = len(bounding_boxes)

    i = 0
    while i < n:
        if not bounding_boxes[i]:

            start = i - 1
            while start >= 0 and not bounding_boxes[start]:
                start -= 1

            end = i
            while (
                end < n
                and not bounding_boxes[end]
                and end - (start if start >= 0 else i) <= max_gap
            ):
                end += 1

            if (
                start >= 0
                and end < n
                and bounding_boxes[start]
                and bounding_boxes[end]
                and end - start <= max_gap
            ):
                start_box = bounding_boxes[start][0]
                end_box = bounding_boxes[end][0]

                for j in range(1, end - start):
                    interp_box = {
                        "class_id": start_box["class_id"],
                        "confidence": min(
                            start_box["confidence"], end_box["confidence"]
                        ),
                        "box": (
                            start_box["box"][0]
                            + (end_box["box"][0] - start_box["box"][0])
                            * j
                            / (end - start),
                            start_box["box"][1]
                            + (end_box["box"][1] - start_box["box"][1])
                            * j
                            / (end - start),
                            start_box["box"][2]
                            + (end_box["box"][2] - start_box["box"][2])
                            * j
                            / (end - start),
                            start_box["box"][3]
                            + (end_box["box"][3] - start_box["box"][3])
                            * j
                            / (end - start),
                        ),
                    }
                    interpolated_bounding_boxes[start + j] = [interp_box]

                i = end
            else:

                i += 1
        else:

            i += 1

    return frames, interpolated_bounding_boxes
