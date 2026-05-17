import cv2
import numpy as np
from ultralytics import YOLO

# -------- LOAD MODEL --------
model = YOLO("yolov8n.pt")

video_path = "../data/test_video_shibuya.mp4"
cap = cv2.VideoCapture(video_path)

people_counts = []
motion_scores = []

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_area = prev_frame.shape[0] * prev_frame.shape[1]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- PERSON DETECTION --------
    results = model(frame, verbose=False)

    count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                count += 1

    people_counts.append(count)

    # -------- MOTION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion = np.mean(magnitude)

    motion_scores.append(motion)

    prev_gray = gray

cap.release()

# -------- METRICS --------
avg_people = np.mean(people_counts)
max_people = np.max(people_counts)

avg_motion = np.mean(motion_scores)
max_motion = np.max(motion_scores)

density = max_people / frame_area * 100000  # scaled

print(f"Avg People: {avg_people:.2f}")
print(f"Max People: {max_people}")
print(f"Avg Motion: {avg_motion:.2f}")
print(f"Max Motion: {max_motion:.2f}")
print(f"Density Score: {density:.2f}")

# -------- WEIGHTED RISK SCORE --------

# Normalize values (important)
people_score = min(max_people / 30, 1.0)     # adjust 30 if needed
motion_score = min(max_motion / 5, 1.0)      # adjust 5 if needed
density_score = min(density / 5, 1.0)        # optional extra signal

# Combine (weights can be tuned)
risk_score = (
    0.5 * people_score +
    0.3 * motion_score +
    0.2 * density_score
)

print(f"Risk Score: {risk_score:.2f}")

# -------- DECISION --------
if risk_score > 0.7:
    print("🚨 HIGH STAMPEDE RISK")
elif risk_score > 0.48:
    print("⚠️ MODERATE RISK (Crowd Building)")
else:
    print("✅ SAFE")