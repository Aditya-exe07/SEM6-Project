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

# -------- LSTM MODEL --------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.dropout = nn.Dropout(0.5)
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

# -------- LOAD VIDEO --------
video_path = "../data/test_video3.mp4"
cap = cv2.VideoCapture(video_path)

frames_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames_list.append(frame)

cap.release()

if len(frames_list) < 20:
    print("❌ Video too short")
    exit()

SEQ_LEN = 20
window_size = SEQ_LEN
step = SEQ_LEN // 2

violence_scores = []

# -------- MOTION PREP --------
prev_gray = cv2.cvtColor(frames_list[0], cv2.COLOR_BGR2GRAY)

# -------- MULTI-WINDOW SCAN --------
for start in range(0, len(frames_list) - window_size, step):

    chunk = frames_list[start:start + window_size]

    sequence = []
    motion_values = []

    for i, frame in enumerate(chunk):

        # -------- MOTION --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        motion_values.append(np.mean(magnitude))

        prev_gray = gray

        # -------- FEATURE --------
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = resnet(img)

        sequence.append(feature.squeeze(0))

    # -------- MODEL PREDICTION --------
    seq_tensor = torch.stack(sequence).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(seq_tensor)
        probs = torch.softmax(output, dim=1)

    violence_prob = probs[0][1].item()
    avg_motion = np.mean(motion_values)

    # -------- COMBINE MOTION --------
    if avg_motion > 2.5:
        violence_scores.append(violence_prob)
    else:
        violence_scores.append(violence_prob * 0.3)

# -------- FINAL DECISION --------
if len(violence_scores) > 0:
    avg_score = sum(violence_scores) / len(violence_scores)
    max_score = max(violence_scores)
else:
    avg_score = 0
    max_score = 0

print(f"Average Violence Score: {avg_score:.2f}")
print(f"Max Violence Score: {max_score:.2f}")

# -------- DECISION LOGIC --------
if max_score > 0.85:
    print("🚨 HIGH VIOLENCE (peak detected)")
elif avg_score > 0.4:
    print("⚠️ SUSTAINED VIOLENCE")
elif max_score > 0.6:
    print("⚠️ POSSIBLE VIOLENCE")
else:
    print("✅ NON-VIOLENCE")