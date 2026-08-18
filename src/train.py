"""
train.py
Beyin tümörü sınıflandırma modelini eğitir.

Kullanım:
    python src/train.py --epochs 15 --batch-size 32 --lr 0.0001
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainTumorDataset, train_transforms, eval_transforms, CLASS_NAMES
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                         help="Sadece son katmanı eğitir (varsayılan: açık)")
    parser.add_argument("--output-dir", type=str, default="models")
    return parser.parse_args()


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Testing")

    train_ds = BrainTumorDataset(train_dir, transform=train_transforms)
    test_ds = BrainTumorDataset(test_dir, transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=args.freeze_backbone).to(device)

    # Malignant (index 0) sınıfı klinik olarak daha kritik ve daha zor
    # ayırt ediliyor; bu sınıfa daha yüksek ağırlık vererek modelin onu
    # kaçırma eğilimini azaltıyoruz. Sıra: [Malignant, Benign, No Tumor]
    class_weights = torch.tensor([1.5, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_loss, val_acc = evaluate(model, test_loader, device, criterion)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"  → Yeni en iyi model kaydedildi (val_acc={val_acc:.4f})")

    print(f"Eğitim tamamlandı. En iyi doğrulama başarımı: {best_acc:.4f}")


if __name__ == "__main__":
    main()
