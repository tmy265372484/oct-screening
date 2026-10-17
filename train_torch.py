import torch, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

train_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
val_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"train"), transform=train_tf)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR,"val"), transform=val_tf)
print(f"类别: {train_ds.classes}")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
best_acc = 0.0
patience = 0
print(f"\n开始训练 {EPOCHS} epoch...")

for epoch in range(EPOCHS):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        tl += loss.item()*imgs.size(0)
        tc += (out.argmax(1)==labels).sum().item()
        tt += labels.size(0)
    train_acc = tc/tt

    model.eval()
    vc, vt, vl = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            vl += criterion(out, labels).item()*imgs.size(0)
            vc += (out.argmax(1)==labels).sum().item()
            vt += labels.size(0)
    val_acc = vc/vt
    scheduler.step(vl/vt)
    print(f"Epoch {epoch+1}/{EPOCHS} | 训练准确率: {train_acc:.4f} | 验证准确率: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  -> 保存最佳模型 {best_acc:.4f}")
        patience = 0
    

print(f"\n完成！最佳验证准确率: {best_acc:.4f}")
