import cv2
import os

def extract_frames(video_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for video_name in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_name)

        # Skip non-video files
        if not video_name.endswith((".mp4", ".avi", ".mov")):
            continue

        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        video_id = video_name.split('.')[0]

        # 🔥 Create folder per video (IMPORTANT)
        video_output_folder = os.path.join(output_folder, video_id)
        os.makedirs(video_output_folder, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save every 5th frame
            if frame_count % 5 == 0:
                filename = f"frame_{frame_count}.jpg"
                filepath = os.path.join(video_output_folder, filename)
                cv2.imwrite(filepath, frame)

            frame_count += 1

        cap.release()
        print(f"Processed {video_name}")

# -------- UPDATED PATHS --------

# Violence videos
extract_frames(
    "../data/Real Life Violence Dataset/Violence",
    "../data/sequences/fight"
)

# Non-violence videos
extract_frames(
    "../data/Real Life Violence Dataset/NonViolence",
    "../data/sequences/non_fight"
)