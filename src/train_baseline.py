import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fma_dataset import FMASpectrogramDataset
from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = FMASpectrogramDataset("data/splits_labeled/train.txt")
val_dataset = FMASpectrogramDataset("data/splits_labeled/val.txt")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")