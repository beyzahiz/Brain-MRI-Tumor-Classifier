import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tensorflow as tf
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import time

# SAYFA AYARLARI 
st.set_page_config(
    page_title="BrainScan AI | Medikal Tanı Ünitesi",
    page_icon="🧠",
    layout="wide"
)

# GELİŞMİŞ TASARIM (Custom CSS) 
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #0B0E14;
        color: #E0E0E0;
    }
    /* Kart Yapıları */
    .metric-card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #58A6FF !important;
        font-family: 'Inter', sans-serif;
    }
    /* Buton Tasarımı */
    .stButton>button {
        background: linear-gradient(45deg, #007bff, #00d4ff);
        color: white;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,123,255,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# FONKSİYONLAR 
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_trained_model():
    model_path = hf_hub_download(repo_id="beyzahiz/brain-mri-tumor-classification", filename="mri_model.keras")
    return tf.keras.models.load_model(model_path)

def medical_preprocessing(img):
    # PIL'den OpenCV formatına güvenli geçiş
    img = np.array(img.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    
    # CLAHE Uygulama
    img_uint8 = img.astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(3):
        img_uint8[:,:,i] = clahe.apply(img_uint8[:,:,i])
    
    img_final = img_uint8.astype('float32')
    return np.expand_dims(img_final, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block7a_project_conv"):
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        base_model = model.layers[0]
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        model = base_model

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = tf.gather(predictions[0], class_idx)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalizasyon ve Threshold
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return np.where(heatmap > 0.2, heatmap, 0)

# ANA PROGRAM 

# Sidebar Tasarımı
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2491/2491214.png", width=80)
    st.title("BrainScan AI")
    st.markdown("---")
    st.info("Bu sistem, **EfficientNetB0** derin öğrenme mimarisini kullanarak MRI görüntülerinde tümör tespiti ve lokalizasyonu yapar.")
    uploaded_file = st.file_uploader("MRI Görüntüsü Yükle", type=["jpg", "png", "jpeg"])
    st.markdown("---")

# Model Yükleme
model = load_trained_model()
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.markdown("# 🧠 Beyin MRI Tümör Analiz Paneli")
st.write("Lütfen sistemin analiz etmesi için sol taraftan bir MRI dosyası seçin.")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader(" Giriş Görüntüsü ")
        st.image(image, use_column_width=True)

    if st.button(" Analizi Başlat ve Lokalize Et "):
        with st.spinner('Yapay zeka katmanları analiz ediliyor. Lütfen bekleyiniz.'):
            time.sleep(1) # Kullanıcı deneyimi için kısa bir bekleme
            
            # 1. Ön İşleme
            processed_img = medical_preprocessing(image)
            
            # 2. Tahmin
            preds = model.predict(processed_img)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx] * 100
            result_label = classes[class_idx]
            
            # 3. Grad-CAM Üretimi
            heatmap = make_gradcam_heatmap(processed_img, model)
            
            # Görselleştirme Hazırlığı
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_cv = cv2.resize(img_cv, (224, 224))
            heatmap_rescaled = np.uint8(255 * heatmap)
            heatmap_rescaled = cv2.resize(heatmap_rescaled, (224, 224))
            heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
            
            with col2:
                st.subheader("📍 Grad-CAM Lokalizasyonu")
                st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # SONUÇ PANELİ (Metric Kartları) 
            st.markdown("---")
            m_col1, m_col2, m_col3 = st.columns(3)
            
            with m_col1:
                st.metric(label="Tahmin Edilen Sınıf", value=result_label)
            with m_col2:
                st.metric(label="Accuracy: ", value=f"%{confidence:.2f}")
            with m_col3:
                status = "Pozitif" if result_label != "No Tumor" else "Negatif"
                st.metric(label="Analiz Durumu", value=status)
            
            if result_label != "No Tumor":
                st.error(f"Dikkat: Görüntüde {result_label} tipi tümör bulgularına rastlanmıştır.")
            else:
                st.success("Herhangi bir tümör bulgusuna rastlanmadı.")

else:
    # Boş durumdayken hoş bir karşılama
    st.write("---")
    st.markdown("""
        ### Nasıl Çalışır?
        1. Sol panelden geçerli bir **MRI görüntüsü** yükleyin.
        2. **Analizi Başlat** butonuna tıklayın.
        3. Modelimiz saniyeler içinde tümörü sınıflandıracak ve **Grad-CAM** teknolojisi ile yerini işaretleyecektir.
    """)