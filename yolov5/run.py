#! /usr/bin/python3

import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or 'yolov5m', 'yolov5l', 'yolov5x'

# Set the path to your input video file
video_path = 'data/videos/hockey1.mp4'
output_path = 'output_video.avi'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties for the output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define a video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference on the frame
    results = model(frame)

    # Render the results on the frame
    frame = results.render()[0]  # results.render() returns a list of rendered images (1 per input)

    # Display the frame
    cv2.imshow('YOLOv5 Detection', frame)

    # Save the frame to the output video
    out.write(frame)

    # Press 'q' to quit the video display window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
