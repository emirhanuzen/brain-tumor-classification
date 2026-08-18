# Veri Seti İndirme Talimatları

1. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset adresinden veri setini indirin.
2. İndirilen klasörü buraya (`data/`) şu yapıda çıkarın:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

3. `src/dataset.py` içindeki `CLASS_MAP` bu klasör isimlerini otomatik olarak
   Benign/Malignant/No Tumor etiketlerine çevirir.

Alternatif olarak Kaggle API ile:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/ --unzip
```
