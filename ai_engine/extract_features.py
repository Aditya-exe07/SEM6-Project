import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD RESNET --------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# -------- TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_path = "../data/sequences"

# 🔥 NEW SAVE STRUCTURE
train_save_path = "../data/features/train"
val_save_path = "../data/features/val"

os.makedirs(train_save_path, exist_ok=True)
os.makedirs(val_save_path, exist_ok=True)

SEQ_LEN = 20

# -------- PROCESS EACH CLASS --------
for label, category in enumerate(["fight", "non_fight"]):

    class_path = os.path.join(data_path, category)

    video_folders = [
    f for f in os.listdir(class_path)
    if os.path.isdir(os.path.join(class_path, f))
]

    # 🔥 SPLIT BEFORE FEATURE EXTRACTION (CRITICAL FIX)
    train_videos, val_videos = train_test_split(
        video_folders, test_size=0.2, random_state=42
    )

    print(f"\nCategory: {category}")
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    # -------- FUNCTION TO PROCESS VIDEOS --------
    def process_videos(video_list, save_path):

        for video_folder in video_list:

            video_path = os.path.join(class_path, video_folder)
            frames = sorted(os.listdir(video_path))

            if len(frames) < SEQ_LEN:
                continue

            # Sample frames across full video
            indices = torch.linspace(0, len(frames)-1, steps=SEQ_LEN).long()
            selected_frames = [frames[i] for i in indices]

            features = []

            for img_name in selected_frames:

                img_path = os.path.join(video_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = resnet(img)

                features.append(feat.squeeze().cpu())

            if len(features) == SEQ_LEN:
                features_tensor = torch.stack(features)

                save_file = os.path.join(
                    save_path, f"{category}_{video_folder}.pt"
                )

                torch.save((features_tensor, label), save_file)

            print(f"Processed {category} → {video_folder}")

    # -------- PROCESS TRAIN --------
    process_videos(train_videos, train_save_path)

    # -------- PROCESS VALIDATION --------
    process_videos(val_videos, val_save_path)

print("\n✅ Feature extraction complete (train + validation split)")