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

# -------- OPTIMIZER --------
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

# -------- LOSS --------
class_weights = torch.tensor([1.0, 1.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -------- PATHS --------
train_path = "../data/features/train"
val_path = "../data/features/val"

train_files = os.listdir(train_path)
val_files = os.listdir(val_path)

print(f"Train samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# -------- TRAINING --------
EPOCHS = 7

for epoch in range(EPOCHS):

    model.train()
    random.shuffle(train_files)

    total_loss = 0
    correct = 0
    total = 0

    # -------- TRAIN LOOP --------
    for file in train_files:

        features, label = torch.load(os.path.join(train_path, file))

        features = features.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        output = model(features)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        _, pred = torch.max(output, 1)
        correct += (pred == label).sum().item()
        total += 1

    avg_loss = total_loss / len(train_files)
    train_acc = 100 * correct / total

    # -------- VALIDATION --------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for file in val_files:

            features, label = torch.load(os.path.join(val_path, file))

            features = features.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)

            output = model(features)
            _, pred = torch.max(output, 1)

            val_correct += (pred == label).sum().item()
            val_total += 1

    val_acc = 100 * val_correct / val_total

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print("-" * 40)

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "../lstm_violence_model.pth")

print("✅ Model training complete and saved.")