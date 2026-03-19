import cv2
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("../violence_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- LOAD VIDEO --------
video_path = "../data/test_video.mp4"

cap = cv2.VideoCapture(video_path)

# 🔥 Smoothing buffers
violence_probs = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- PREPROCESS --------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0).to(device)

    # -------- PREDICTION --------
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

    # Probability of violence
    violence_prob = probs[0][1].item()

    # -------- SMOOTHING --------
    violence_probs.append(violence_prob)

    if len(violence_probs) > 30:   # increased window
        violence_probs.pop(0)

    avg_prob = sum(violence_probs) / len(violence_probs)

    # -------- DECISION --------
    if avg_prob > 0.4:   # lowered threshold
        label = "Violence"
        color = (0,0,255)
    else:
        label = "Non-Violence"
        color = (0,255,0)

    # -------- DISPLAY --------
    cv2.putText(
        frame,
        f"{label}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.putText(
        frame,
        f"Violence Prob: {avg_prob:.2f}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.imshow("Violence Detection Test", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()