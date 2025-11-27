# ============================================
# Force old keras loader
# ============================================
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# ============================================
# Disease configuration dictionary
# ============================================
DISEASES = {
    "Brain Tumor MRI": {
        "MODEL_PATH": "MRI_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["Glioma", "Meningioma", "No_Tumor", "Pituitary"],
        "ACCURACY": 0.9,
        "TITLE": "🧠 Brain Tumor MRI Classification",
        "DESC": "Upload an MRI image to classify tumor type using the trained CNN model."
    },
    "Gallbladder Ultrasound": {
        "MODEL_PATH": "Ultrasound_gallblader_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["Abdomen", "Adenomyoma", "Carcinoma", "Cholecystitis", "Gallstones", "Gangrenous", "Perforation", "Polyps", "WallThickening"],
        "ACCURACY": 0.998,
        "TITLE": "🫁 Gallbladder Ultrasound Classification",
        "DESC": "Upload a gallbladder ultrasound image to classify conditions."
    },
    "Lung Cancer": {
        "MODEL_PATH": "Lung_best_model.h5",
        "IMG_SIZE": (224, 224),
        "CLASS_NAMES": ["lung_aca", "lung_scc", "lung_n"],
        "ACCURACY": 0.99,
        "TITLE": "🌿 Lung Cancer Classification",
        "DESC": "Upload a lung image to classify cancer type."
    }
}


# ============================================
# Load model cached
# ============================================
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


# ============================================
# Preprocess image
# ============================================
def preprocess_image(img, img_size):
    img = img.resize(img_size)
    img = np.array(img) / 255.0
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = np.expand_dims(img, axis=0)
    return img


# ============================================
# Generate PDF Report
# ============================================
# def generate_pdf(disease, result_class, confidence):
#     file_path = "/mnt/data/medical_report.pdf"
#     c = canvas.Canvas(file_path, pagesize=letter)
    
#     c.setFont("Helvetica-Bold", 20)
#     c.drawString(50, 750, "Medical AI Diagnosis Report")

#     c.setFont("Helvetica", 12)
#     c.drawString(50, 720, f"Disease Type: {disease}")
#     c.drawString(50, 700, f"Prediction Result: {result_class.upper()}")
#     c.drawString(50, 680, f"Confidence: {confidence*100:.2f}%")
#     c.drawString(50, 660, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#     c.save()
#     return file_path

# ============================================
# Generate PDF Report
# ============================================
def generate_pdf(disease, result_class, confidence):
    import tempfile
    
    # Use temporary directory or current directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "medical_report.pdf")
    
    c = canvas.Canvas(file_path, pagesize=letter)
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 750, "Medical AI Diagnosis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Disease Type: {disease}")
    c.drawString(50, 700, f"Prediction Result: {result_class.upper()}")
    c.drawString(50, 680, f"Confidence: {confidence*100:.2f}%")
    c.drawString(50, 660, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.save()
    return file_path

# ============================================
# Streamlit page config
# ============================================
st.set_page_config(page_title="Medical AI", layout="wide")

# History storage
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================
# Sidebar
# ============================================
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to:", ["🩺 Diagnose", "📜 History Log", "📊 Model Info"])

disease = st.sidebar.selectbox("🔍 Select Disease Model", list(DISEASES.keys()))
config = DISEASES[disease]
model = load_model(config["MODEL_PATH"])


# ============================================
# PAGE 1: DIAGNOSE
# ============================================
if page == "🩺 Diagnose":

    # Center content area
    st.markdown(
        """
        <style>
        .centered-container {
            max-width: 750px;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown("<div class='centered-container'>", unsafe_allow_html=True)

        st.markdown(f"## {config['TITLE']}")
        st.write(config["DESC"])

        uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

        if uploaded_file:

            st.image(uploaded_file, caption="Uploaded Image", width=400)

            img = Image.open(uploaded_file)
            processed_img = preprocess_image(img, config["IMG_SIZE"])

            # Progress
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)
                time.sleep(0.01)

            prediction = model.predict(processed_img)
            pred_idx = np.argmax(prediction)
            confidence = float(np.max(prediction))
            result_class = config["CLASS_NAMES"][pred_idx]

            # Save history log
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disease": disease,
                "result": result_class,
                "confidence": confidence,
                "file": uploaded_file.name
            })

            st.success(f"### 🧾 Prediction: **{result_class.upper()}** ({confidence*100:.2f}%)")

            # # Confidence chart
            # st.subheader("📈 Confidence Chart")
            # fig, ax = plt.subplots(figsize=(6, 3))
            # bars = ax.bar(config["CLASS_NAMES"], prediction[0], color='#667eea')
            # ax.set_ylabel('Confidence')
            # ax.set_ylim([0, 1])
            # ax.grid(axis='y', alpha=0.3)
            # plt.xticks(rotation=45, ha='right')
            # plt.tight_layout()
            # st.pyplot(fig, use_container_width=False)
            # plt.close(fig)

            # PDF report
            pdf_path = generate_pdf(disease, result_class, confidence)
            st.download_button(
                "📄 Download PDF Report",
                open(pdf_path, "rb"),
                file_name="medical_report.pdf",
                mime="application/pdf"
            )

        st.markdown("</div>", unsafe_allow_html=True)




# ============================================
# PAGE 2: HISTORY LOG
# ============================================
elif page == "📜 History Log":
    st.header("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No history available yet.")
    else:
        for entry in st.session_state.history:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 15px;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <b style="font-size:16px;">⏰ {entry['time']}</b><br><br>
                    <b>🏥 Disease:</b> {entry['disease']}<br>
                    <b>🎯 Result:</b> {entry['result'].upper()}<br>
                    <b>📊 Confidence:</b> {entry['confidence']*100:.2f}%<br>
                    <b>📁 File:</b> {entry['file']}
                </div>
                """,
                unsafe_allow_html=True
            )

# ============================================
# PAGE 3: MODEL INFO
# ============================================
elif page == "📊 Model Info":
    st.header("📊 Model Information")

    st.write(f"### Model for {disease}")
    st.write(f"**Image Size:** {config['IMG_SIZE']}")
    st.write(f"**Classes:** {', '.join(config['CLASS_NAMES'])}")
    st.write(f"**Accuracy:** {config['ACCURACY']*100:.2f}%")
