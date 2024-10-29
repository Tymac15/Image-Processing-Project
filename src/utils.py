import cv2


def export_frames_to_video(frames, output_path="output_video.mp4", fps=25):

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")
