import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_path = "../data/sequences"
save_path = "../data/features"

os.makedirs(save_path, exist_ok=True)

SEQ_LEN = 20   # 🔥 increased sequence length

for label, category in enumerate(["fight", "non_fight"]):

    class_path = os.path.join(data_path, category)

    for video_folder in os.listdir(class_path):

        video_path = os.path.join(class_path, video_folder)

        frames = sorted(os.listdir(video_path))

        # 🔥 Skip very short videos
        if len(frames) < SEQ_LEN:
            continue

        # 🔥 SAMPLE FRAMES ACROSS WHOLE VIDEO (CRITICAL FIX)
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

        # Save only if correct length
        if len(features) == SEQ_LEN:
            features_tensor = torch.stack(features)

            save_file = os.path.join(save_path, f"{category}_{video_folder}.pt")
            torch.save((features_tensor, label), save_file)

        print(f"Processed {video_folder}")