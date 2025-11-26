# Force TensorFlow to load old .h5 format correctly
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =============================
# CONFIG
# =============================
MODEL_PATH = "mri_best_model.h5"
IMG_SIZE = (224, 224)  # <-- change to your model input size
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]  
# Change based on your dataset labels


# =============================
# Load Model Once
# =============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()


# =============================
# Preprocess Image
# =============================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0

    if img.shape[-1] == 4:  # handle PNG with alpha channel
        img = img[..., :3]

    img = np.expand_dims(img, axis=0)
    return img


# =============================
# Streamlit UI
# =============================
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify tumor type using your CNN model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = Image.open(uploaded_file)
    processed_img = preprocess_image(img)

    st.write("### 🔄 Making Prediction...")
    prediction = model.predict(processed_img)
    pred_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    result_class = CLASS_NAMES[pred_index]

    st.success(f"### 🧾 Prediction: **{result_class.upper()}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

