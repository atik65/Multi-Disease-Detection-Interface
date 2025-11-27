# ============================================
# Force TensorFlow to load old .h5 format
# ============================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================
# Disease Model Configurations (Dynamic)
# ============================================
DISEASES = {
    "Brain Tumor": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "TITLE": "🧠 Brain Tumor MRI Classification",
        "DESC": "Upload an MRI image to classify tumor type using the trained CNN model."
    },

    "Pneumonia": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["normal", "pneumonia"],
        "TITLE": "🫁 Pneumonia X-Ray Classification",
        "DESC": "Upload a chest X-ray image to detect pneumonia."
    },

    "Skin Cancer": {
        "MODEL_PATH": "mri_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["benign", "malignant"],
        "TITLE": "🌿 Skin Cancer Classification",
        "DESC": "Upload a skin lesion image to classify cancer type."
    }
}

# ============================================
# Load Model (Cached) - Will reload per disease selection
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

    if img.shape[-1] == 4:  # PNG with alpha
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)
    return img


# ============================================
# Modern Streamlit UI
# ============================================
st.set_page_config(page_title="Medical AI Diagnostic", layout="centered")

# Stylish header
st.markdown(
    """
    <h1 style='text-align:center; color:#4A90E2;'>🧬 AI Medical Disease Classifier</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
    Select a disease and upload an image for AI-based diagnosis.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Disease selector
disease = st.selectbox(
    "Select Disease Type",
    list(DISEASES.keys()),
    index=0
)

config = DISEASES[disease]

st.title(config["TITLE"])
st.write(config["DESC"])

# Load selected model
model = load_model(config["MODEL_PATH"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img, config["IMG_SIZE"])

    st.write("### 🔄 Making Prediction...")
    prediction = model.predict(processed_img)
    pred_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    result_class = config["CLASS_NAMES"][pred_idx]

    st.success(f"### 🧾 Prediction: **{result_class.upper()}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
