import streamlit as st
import cv2
import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import tempfile

# ------------------ SETUP ------------------
st.set_page_config(page_title="Violence Detection", layout="wide")

st.title("🚨 Violence Detection System")
st.write("Upload a video to analyze potential violent activity.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

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
    model.load_state_dict(torch.load("lstm_violence_model.pth", map_location=device))
    model.eval()

    return resnet, model

resnet, model = load_models()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

SEQ_LEN = 20

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    if st.button("🔍 Analyze Video"):

        cap = cv2.VideoCapture(tfile.name)

        sequence = []
        violence_scores = []

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        # ---- INIT FIRST FRAME ----
        ret, prev_frame = cap.read()
        if not ret:
            st.error("Error reading video")
            st.stop()

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # ------------------ PROCESS VIDEO ------------------
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1
            progress_bar.progress(min(processed_frames / frame_count, 1.0))

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

                # Motion weighting
                if motion_score > 2.5:
                    violence_scores.append(violence_prob)
                else:
                    violence_scores.append(violence_prob * 0.3)

        cap.release()

        # ------------------ FINAL RESULT ------------------
        if len(violence_scores) > 0:
            avg_score = sum(violence_scores) / len(violence_scores)
            max_score = max(violence_scores)
        else:
            avg_score = 0
            max_score = 0

        st.subheader("📊 Results")
        st.write(f"**Average Score:** {avg_score:.2f}")
        st.write(f"**Max Score:** {max_score:.2f}")

        # Decision logic
        if max_score > 0.85 and avg_score > 0.2:
            st.error("🚨 HIGH VIOLENCE DETECTED")
        elif max_score > 0.7 and avg_score > 0.15:
            st.warning("⚠️ POSSIBLE VIOLENCE")
        elif avg_score > 0.35:
            st.warning("⚠️ SUSTAINED ACTIVITY")
        else:
            st.success("✅ NON-VIOLENCE")