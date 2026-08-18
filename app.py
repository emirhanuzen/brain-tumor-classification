"""
app.py
Beyin tümörü MRI sınıflandırma modeli için Streamlit web arayüzü.

Kullanım:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# set_page_config, diğer tüm Streamlit çağrılarından önce gelmelidir.
st.set_page_config(layout="wide",
                   page_title="Beyin Tümörü Sınıflandırması",
                   page_icon="🧠")

# src/ klasöründeki modüller birbirini düz isimle (from dataset import ...)
# içe aktardığı için bu klasörü arama yoluna ekliyoruz.
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import CLASS_NAMES, eval_transforms  # noqa: E402
from model import build_model  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "best_model.pth")

# Model sınıf adlarının arayüzde gösterilecek Türkçe karşılıkları.
# Sıra dataset.py içindeki CLASS_NAMES ile birebir aynıdır.
CLASS_LABELS_TR = {
    "Malignant": "Kötü Huylu (Malignant)",
    "Benign": "İyi Huylu (Benign)",
    "No Tumor": "Tümör Yok (No Tumor)",
}


@st.cache_resource
def load_model():
    """Modeli bir kez oluşturup ağırlıkları yükler ve önbelleğe alır.

    @st.cache_resource sayesinde her "Tahmin Et" tıklamasında model yeniden
    yüklenmez; ilk çalıştırmada bir kez yüklenip bellekte tutulur.
    """
    # pretrained=False: ImageNet ağırlıkları best_model.pth ile üzerine yazılacağı
    # için indirilmelerine gerek yok (uygulama internetsiz de açılabilir).
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    # map_location="cpu": kullanıcıda GPU olmayabilir, ağırlıklar CPU'ya yüklenir.
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()  # Dropout/BatchNorm katmanlarını çıkarım moduna alır
    return model


def predict(model, image):
    """Bir PIL görüntüsü için sınıf olasılıklarını döndürür.

    Returns:
        (en_yuksek_sinif_index, olasiliklar_listesi)
    """
    # Eğitimde kullanılan ön işleme adımlarının aynısı (dataset.py -> eval_transforms)
    tensor = eval_transforms(image.convert("RGB")).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    return int(probs.argmax().item()), probs.tolist()


# --- Başlık ve açıklama --------------------------------------------------

st.title("🧠 Beyin Tümörü Sınıflandırması")

st.markdown(
    """
    Bu araç, beyin MRI görüntülerini **Kötü Huylu (Malignant)**, **İyi Huylu (Benign)**
    ve **Tümör Yok (No Tumor)** olmak üzere üç sınıfa ayıran, ResNet18 tabanlı
    transfer learning modelini kullanır.

    Proje, **Microsoft AI Innovator** programı kapsamında geliştirilmiştir.
    """
)

st.warning(
    "⚠️ **Bu uygulama klinik tanı amaçlı değildir.** Yalnızca eğitim ve araştırma "
    "amacıyla geliştirilmiştir; tıbbi karar verme sürecinde kullanılamaz."
)

st.divider()

# --- Görüntü yükleme -----------------------------------------------------

uploaded_file = st.file_uploader(
    "Bir MRI görüntüsü yükleyin (JPG veya PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Sol sütun: yüklenen görüntü + Tahmin Et butonu
    # Sağ sütun: tahmin sonucu ve olasılık grafiği
    col_image, col_result = st.columns(2)

    with col_image:
        st.image(image, caption="Yüklenen MRI görüntüsü", width=350)
        run_prediction = st.button("🔍 Tahmin Et", type="primary", width="stretch")

    # --- Tahmin ----------------------------------------------------------
    if run_prediction:
        with col_result:
            if not os.path.exists(MODEL_PATH):
                st.error(
                    f"Model dosyası bulunamadı: `{MODEL_PATH}`\n\n"
                    "Önce `python src/train.py` ile modeli eğitmeniz gerekiyor."
                )
            else:
                with st.spinner("Model çalışıyor, lütfen bekleyin..."):
                    model = load_model()
                    pred_index, probs = predict(model, image)

                pred_class = CLASS_NAMES[pred_index]
                pred_label = CLASS_LABELS_TR[pred_class]
                confidence = probs[pred_index] * 100

                st.subheader("Sonuç")

                # Renk kodlaması: kötü huylu tahminler kırmızı uyarı kutusunda,
                # diğerleri yeşil kutuda gösterilir.
                result_text = f"### {pred_label}\n**Güven: %{confidence:.1f}**"
                if pred_class == "Malignant":
                    st.error(result_text)
                else:
                    st.success(result_text)

                # --- Olasılık dağılımı ---------------------------------------
                st.markdown("#### Sınıf Olasılıkları")

                prob_df = pd.DataFrame(
                    {"Olasılık (%)": [p * 100 for p in probs]},
                    index=[CLASS_LABELS_TR[name] for name in CLASS_NAMES],
                )
                st.bar_chart(prob_df)

                # Sayısal değerleri de tablo olarak gösteriyoruz
                st.dataframe(
                    prob_df.style.format({"Olasılık (%)": "{:.2f}"}),
                    width="stretch",
                )
else:
    st.info("Tahmin yapmak için yukarıdan bir MRI görüntüsü yükleyin.")

# --- Sorumluluk reddi ----------------------------------------------------

st.divider()

st.caption(
    "**Sorumluluk Reddi:** Bu araç yalnızca eğitim ve araştırma amaçlıdır. "
    "Klinik tanı, tedavi planlaması veya herhangi bir tıbbi karar için "
    "kullanılamaz. Model çıktıları hatalı olabilir; sağlık sorunlarınız için "
    "mutlaka bir hekime başvurun."
)
