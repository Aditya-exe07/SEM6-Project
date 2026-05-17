import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- CONFIG --------
DATA_PATH = "../data/videos"
CLIP_LEN = 16        # faster training
IMG_SIZE = 112       # faster training
BATCH_SIZE = 4
EPOCHS = 5

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])

# -------- LOAD VIDEO CLIP --------
def load_clip(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        frames.append(frame)

    cap.release()

    if len(frames) < CLIP_LEN:
        return None

    try:
        indices = np.linspace(0, len(frames) - 1, CLIP_LEN).astype(int)

        clip = []
        for i in indices:
            img = transform(frames[i])
            clip.append(img)

        clip = torch.stack(clip)          # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)   # (C, T, H, W)

        return clip

    except:
        return None

# -------- DATASET --------
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        clip = load_clip(path)

        if clip is None:
            for _ in range(10):
                path, label = random.choice(self.samples)
                clip = load_clip(path)
                if clip is not None:
                    return clip, label

            # fallback (rare)
            return torch.zeros(3, CLIP_LEN, IMG_SIZE, IMG_SIZE), 0

        return clip, label

# -------- PREPARE DATA --------
all_samples = []

for label, category in enumerate(["non_violence", "violence"]):
    class_path = os.path.join(DATA_PATH, category)

    for video in os.listdir(class_path):

        # 🔥 IMPORTANT: filter only video files
        if not video.endswith((".mp4", ".avi", ".mov")):
            continue

        path = os.path.join(class_path, video)
        all_samples.append((path, label))

# -------- TRAIN / VAL SPLIT --------
train_samples, val_samples = train_test_split(
    all_samples, test_size=0.2, random_state=42
)

print(f"Train samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

train_dataset = VideoDataset(train_samples)
val_dataset = VideoDataset(val_samples)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# -------- MODEL --------
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# -------- TRAIN --------
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for clips, labels in train_loader:

        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for clips, labels in val_loader:

            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            _, pred = torch.max(outputs, 1)

            val_correct += (pred == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-" * 40)

# -------- SAVE --------
torch.save(model.state_dict(), "../r3d_violence_model.pth")

print("✅ 3D CNN model trained successfully!")