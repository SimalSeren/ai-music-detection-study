import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from cropped_dataset import CroppedSpectrogramDataset
from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = CroppedSpectrogramDataset(
    "data/splits_labeled/train.txt",
    crop_width=64,
    stride=32
)

val_dataset = CroppedSpectrogramDataset(
    "data/splits_labeled/val.txt",
    crop_width=64,
    stride=32
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for x, y, _, _ in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total += y.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, _, _ in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_targets, all_preds)
    val_prec = precision_score(all_targets, all_preds, zero_division=0)
    val_rec = recall_score(all_targets, all_preds, zero_division=0)
    val_f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}"
    )
    print("Confusion Matrix:\n", cm)