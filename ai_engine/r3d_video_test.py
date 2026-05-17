import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_LEN = 16
IMG_SIZE = 112

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])

# -------- MODEL --------
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("../r3d_violence_model.pth", map_location=device))
model = model.to(device)
model.eval()

# -------- LOAD VIDEO --------
video_path = "../data/test_video3.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

if len(frames) < CLIP_LEN:
    print("Video too short")
    exit()

# -------- MULTI-CLIP INFERENCE --------
step = CLIP_LEN // 2
scores = []

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
        output = model(clip)
        probs = torch.softmax(output, dim=1)

    scores.append(probs[0][1].item())  # violence prob

# -------- FINAL RESULT --------
avg_score = np.mean(scores)
max_score = np.max(scores)

print(f"Average Violence Score: {avg_score:.2f}")
print(f"Max Violence Score: {max_score:.2f}")

if max_score > 0.85:
    print("🚨 HIGH VIOLENCE")
elif avg_score > 0.5:
    print("⚠️ POSSIBLE VIOLENCE")
else:
    print("✅ NON-VIOLENCE")