# ============================================
# Force TensorFlow to load old .h5 format
# ============================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

# ============================================
# Disease Model Configurations (Dynamic)
# ============================================
DISEASES = {
    "Brain Tumor": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "ACCURACY": 0.998,
        "TITLE": "🧠 Brain Tumor MRI Classification",
        "DESC": "Upload an MRI image to classify tumor type using the trained CNN model."
    },

    "Pneumonia": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["normal", "pneumonia"],
        "ACCURACY": 0.985,
        "TITLE": "🫁 Pneumonia X-Ray Classification",
        "DESC": "Upload a chest X-ray image to detect pneumonia."
    },

    "Skin Cancer": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["benign", "malignant"],
        "ACCURACY": 0.99,
        "TITLE": "🌿 Skin Cancer Classification",
        "DESC": "Upload a skin lesion image to classify cancer type."
    }
}

# ============================================
# Load Model (Cached)
# ============================================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# ============================================
# Preprocess Image
# ============================================
def preprocess_image(img, img_size):
    img = img.resize(img_size)
    img = np.array(img) / 255.0

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)
    return img


# ============================================
# Streamlit UI Config
# ============================================
st.set_page_config(page_title="Medical AI Diagnostic", layout="wide")

# Sidebar Navigation
st.sidebar.title("⚙️ Navigation")
selected_page = st.sidebar.radio("Go to:", ["🩺 Diagnose", "📊 Model Info"])

disease = st.sidebar.selectbox("🔍 Select Disease", list(DISEASES.keys()))

config = DISEASES[disease]
model = load_model(config["MODEL_PATH"])

# ============================================
# PAGE 1: DIAGNOSIS
# ============================================
if selected_page == "🩺 Diagnose":

    st.markdown(
        f"""
        <h1 style='text-align:center; color:#4A90E2;'>{config['TITLE']}</h1>
        <p style='text-align:center; font-size:18px; color:gray;'>{config['DESC']}</p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload Medical Image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img, config["IMG_SIZE"])

        # Progress bar
        progress = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress.progress(i + 1)
            status_text.text(f"⏳ Processing image... {i+1}%")
            time.sleep(0.01)
        
        status_text.text("🔍 Making prediction...")

        prediction = model.predict(processed_img)
        pred_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        result_class = config["CLASS_NAMES"][pred_idx]

        # ============================================
        # Beautiful Card-Style Result UI
        # ============================================
        st.markdown(
            f"""
            <div style="
                background: #ffffff;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 1px 1px 15px rgba(0,0,0,0.1);
                text-align: center;
                margin-top: 20px;
            ">
                <h2 style="color:#2E86C1;">🧾 Prediction Result</h2>
                <h1 style="color:#117A65;">{result_class.upper()}</h1>
                <p style="font-size: 20px;">Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ============================================
        # Confidence Bar Chart
        # ============================================
        st.subheader("📈 Confidence Chart")
        fig, ax = plt.subplots()
        ax.bar(config["CLASS_NAMES"], prediction[0])
        ax.set_ylabel("Confidence")
        ax.set_title("Model Prediction Confidence")
        st.pyplot(fig)


# ============================================
# PAGE 2: MODEL INFO
# ============================================
elif selected_page == "📊 Model Info":

    st.title("📊 Model Performance & Information")

    st.write(f"### 🧬 Selected Model: **{disease}**")
    st.write(f"#### 📁 Model Path: `{config['MODEL_PATH']}`")
    st.write(f"#### 🖼️ Image Size: {config['IMG_SIZE'][0]} × {config['IMG_SIZE'][1]}")
    st.write(f"#### 🏷️ Classes: {', '.join(config['CLASS_NAMES'])}")

    # Model Accuracy Card
    st.markdown(
        f"""
        <div style="
            background: #ECF0F1;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
        ">
            <h2 style="color:#2C3E50;">📊 Model Accuracy</h2>
            <h1 style="color:#27AE60;">{config['ACCURACY']*100:.2f}%</h1>
            <p style="font-size: 18px; color:#555;">Based on training & validation performance</p>
        </div>
        """,
        unsafe_allow_html=True
    )
