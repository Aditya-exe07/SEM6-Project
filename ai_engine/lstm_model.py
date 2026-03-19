import torch
import torch.nn as nn
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- MODEL --------
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

# 🔥 LOWER LR (more stable)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

# 🔥 CLASS WEIGHTS (CRITICAL FIX)
class_weights = torch.tensor([1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

feature_path = "../data/features"
files = os.listdir(feature_path)

# 🔥 OPTIONAL: balance dataset manually
fight_files = [f for f in files if "fight" in f]
non_fight_files = [f for f in files if "non_fight" in f]

min_len = min(len(fight_files), len(non_fight_files))
files = fight_files[:min_len] + non_fight_files[:min_len]

print(f"Training on {len(files)} samples (balanced)")

# -------- TRAINING --------
for epoch in range(5):

    random.shuffle(files)
    total_loss = 0

    correct = 0
    total = 0

    for file in files:

        features, label = torch.load(os.path.join(feature_path, file))

        features = features.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        output = model(features)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()

        # 🔥 Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        # 🔥 Track accuracy during training
        _, pred = torch.max(output, 1)
        correct += (pred == label).sum().item()
        total += 1

    avg_loss = total_loss / len(files)
    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "../lstm_violence_model.pth")