from ultralytics import YOLO


def run_yolo_inference(video_path, model_path="yolov8s.pt",  conf=0.1):

    model = YOLO(model_path)

    results = model.predict(source=video_path, save=False, device="cuda", conf=conf)

    frames = []
    bounding_boxes = []

    for result in results:
        frame = result.orig_img
        boxes = result.boxes

        frame_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            frame_boxes.append(
                {
                    "class_id": class_id,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                }
            )

        frames.append(frame)
        bounding_boxes.append(frame_boxes)

    return frames, bounding_boxes
