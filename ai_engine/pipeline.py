import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from ultralytics import YOLO
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD VIOLENCE MODEL --------
weights = R3D_18_Weights.DEFAULT
violence_model = r3d_18(weights=weights)
violence_model.fc = nn.Linear(violence_model.fc.in_features, 2)
violence_model.load_state_dict(torch.load("../r3d_violence_model.pth", map_location=device))
violence_model = violence_model.to(device)
violence_model.eval()

# -------- YOLO --------
yolo_model = YOLO("yolov8n.pt")

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])

CLIP_LEN = 16


def run_pipeline(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    people_counts = []
    motion_scores = []

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_area = prev_frame.shape[0] * prev_frame.shape[1]

    # -------- READ VIDEO --------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        # -------- YOLO PEOPLE COUNT --------
        results = yolo_model(frame, verbose=False)
        count = sum(int(box.cls[0]) == 0 for r in results for box in r.boxes)
        people_counts.append(count)

        # -------- MOTION --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_scores.append(np.mean(magnitude))

        prev_gray = gray

    cap.release()

    # -------- STAMPEDE ANALYSIS --------
    max_people = np.max(people_counts)
    max_motion = np.max(motion_scores)

    density = max_people / frame_area * 100000

    people_score = min(max_people / 30, 1.0)
    motion_score = min(max_motion / 5, 1.0)
    density_score = min(density / 5, 1.0)

    stampede_score = (
        0.5 * people_score +
        0.3 * motion_score +
        0.2 * density_score
    )

    # -------- VIOLENCE ANALYSIS --------
    step = CLIP_LEN // 2
    violence_scores = []

    for start in range(0, len(frames) - CLIP_LEN, step):

        clip_frames = frames[start:start + CLIP_LEN]
        clip = []

        for frame in clip_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)
            clip.append(img)

        clip = torch.stack(clip).permute(1, 0, 2, 3).unsqueeze(0).to(device)

        with torch.no_grad():
            output = violence_model(clip)
            probs = torch.softmax(output, dim=1)

        violence_scores.append(probs[0][1].item())

    max_violence = max(violence_scores) if violence_scores else 0

    return {
        "violence_score": max_violence,
        "stampede_score": stampede_score,
        "max_people": int(max_people)
    }