"""
evaluate.py
Eğitilmiş modeli test seti üzerinde değerlendirir; confusion matrix ve
sınıflandırma raporu üretir.

Kullanım:
    python src/evaluate.py --model-path models/best_model.pth
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import BrainTumorDataset, eval_transforms, CLASS_NAMES
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = os.path.join(args.data_dir, "Testing")
    test_ds = BrainTumorDataset(test_dir, transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(out_path)
    print(f"Confusion matrix kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
