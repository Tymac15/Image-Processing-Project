#! /usr/bin/python3

import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'runs\train\run_22\weights\best.pt', force_reload=True, trust_repo=True)


video_path = 'data/videos/hockey0.mp4'
output_path = '11_Oct.avi'

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)

    frame = results.render()[0]

    cv2.imshow('YOLOv5 Detection', frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
