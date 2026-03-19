import cv2
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD RESNET --------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# -------- LSTM MODEL (MATCH TRAINING) --------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)   # 🔥 MUST MATCH TRAINING
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        return out

model = LSTMModel().to(device)
model.load_state_dict(torch.load("../lstm_violence_model.pth", map_location=device))
model.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- VIDEO --------
video_path = "../data/test_video1.mp4"
cap = cv2.VideoCapture(video_path)

sequence = []
SEQ_LEN = 20   # 🔥 match training

violence_scores = []

# 🔥 Motion tracking (important)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- MOTION --------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion_score = np.mean(magnitude)

    prev_gray = gray

    # -------- PREPROCESS --------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # -------- FEATURE --------
    with torch.no_grad():
        feature = resnet(img)

    sequence.append(feature.squeeze(0))

    if len(sequence) > SEQ_LEN:
        sequence.pop(0)

    # -------- PREDICTION --------
    if len(sequence) == SEQ_LEN:
        seq_tensor = torch.stack(sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(seq_tensor)
            probs = torch.softmax(output, dim=1)

        violence_prob = probs[0][1].item()

        # 🔥 COMBINE WITH MOTION (CRITICAL FIX)
        if motion_score > 2.5:
            violence_scores.append(violence_prob)
        else:
            violence_scores.append(violence_prob * 0.3)  # dampen false positives

# -------- FINAL DECISION --------
cap.release()

if len(violence_scores) > 0:
    avg_score = sum(violence_scores) / len(violence_scores)
    max_score = max(violence_scores)
else:
    avg_score = 0
    max_score = 0

print(f"Average Violence Score: {avg_score:.2f}")
print(f"Max Violence Score: {max_score:.2f}")

# 🔥 IMPROVED DECISION LOGIC
if max_score > 0.85 and avg_score > 0.2:
    print("🚨 HIGH VIOLENCE DETECTED (strong peak)")
elif max_score > 0.7 and avg_score > 0.15:
    print("⚠️ POSSIBLE VIOLENCE")
elif avg_score > 0.35:
    print("⚠️ SUSTAINED ACTIVITY")
else:
    print("✅ NON-VIOLENCE")