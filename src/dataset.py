"""
dataset.py
Beyin MRI görüntülerini yükler ve orijinal Kaggle sınıflarını
(glioma, meningioma, pituitary, notumor) İyi Huylu / Kötü Huylu / Tümör Yok
etiketlerine eşler.

Tıbbi not: Bu eşleme genel literatür eğilimlerine dayanan bir basitleştirmedir,
klinik tanı yerine geçmez (bkz. README.md).
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Orijinal klasör adı -> yeni sınıf indexi
CLASS_MAP = {
    "glioma": 0,       # Kötü Huylu (Malignant)
    "meningioma": 1,   # İyi Huylu (Benign)
    "pituitary": 1,    # İyi Huylu (Benign)
    "notumor": 2,      # Tümör Yok
}

CLASS_NAMES = ["Malignant", "Benign", "No Tumor"]

IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class BrainTumorDataset(Dataset):
    """Kaggle Brain Tumor MRI klasör yapısını okuyup Benign/Malignant/No Tumor
    etiketleriyle bir PyTorch Dataset'i oluşturur."""

    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for folder_name, label in CLASS_MAP.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"'{root_dir}' içinde görüntü bulunamadı. "
                f"Lütfen data/README.md içindeki indirme talimatlarını takip edin."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
