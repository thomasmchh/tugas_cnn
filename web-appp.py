import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

# ================= CONFIG =================
st.set_page_config(page_title="Klasifikasi Kuda vs Manusia", layout="wide")

st.title("Klasifikasi Kuda vs Manusia ")

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/hasil_model.h5')
    return model

model = load_model()

# ================= HISTORY =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= UPLOAD =================
uploaded_files = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")

        # ================= PREPROCESS =================
        img_bytes = uploaded_file.getvalue()
        img = Image.open(BytesIO(img_bytes)).convert("RGB").resize((150, 150))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ================= PREDICT =================
        with st.spinner("Memproses..."):
            prediction = model.predict(img_array)

        confidence = prediction[0][0]

        # ================= THRESHOLD =================
        if 0.4 < confidence < 0.6:
            predicted_class = "Tidak Dikenali"
            conf_percent = max(confidence, 1 - confidence) * 100
        elif confidence >= 0.6:
            predicted_class = "Manusia"
            conf_percent = confidence * 100
        else:
            predicted_class = "Kuda"
            conf_percent = (1 - confidence) * 100

        prob_manusia = confidence
        prob_kuda = 1 - confidence

        # ================= SAVE HISTORY =================
        st.session_state.history.append({
            "nama": uploaded_file.name,
            "label": predicted_class,
            "confidence": conf_percent
        })

        # ================= DISPLAY =================
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption=uploaded_file.name)

        with col2:
            st.subheader("Hasil Prediksi")
            st.write(f"**Label:** {predicted_class}")
            st.write(f"**Confidence:** {conf_percent:.2f}%")

            st.write("### Probabilitas")
            st.progress(float(prob_manusia))
            st.write(f"Manusia: {prob_manusia*100:.2f}%")

            st.progress(float(prob_kuda))
            st.write(f"Kuda: {prob_kuda*100:.2f}%")

        # ================= WARNING =================
        if predicted_class == "Tidak Dikenali":
            st.warning("Model tidak yakin. Gambar mungkin bukan kuda atau manusia.")

# ================= HISTORY =================
st.markdown("## History Prediksi")

if st.session_state.history:
    for item in st.session_state.history:
        st.write(f"{item['nama']} → {item['label']} ({item['confidence']:.2f}%)")

    if st.button("Hapus History"):
        st.session_state.history = []
else:
    st.info("Belum ada history")

# ================= FOOTER =================
st.markdown("---")
st.caption("CNN + Transfer Learning + Multi Upload + Confidence Score")