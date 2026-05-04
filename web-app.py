# Running: streamlit run .\web-app.py
# Python version 3.12

import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Klasifikasi Kuda dan Manusia")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

def load_cnn_model():
    model = tf.keras.models.load_model('model/hasil_model.h5')
    return model

model = load_cnn_model()

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    img = Image.open(BytesIO(img_bytes)).resize((150, 150))
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        predicted_class = 'Manuia'
    else:
        predicted_class = 'Kuda'

    st.image(img, use_column_width=False)
    st.write(f"**Prediksi:** {predicted_class}")
else:
    st.write("Belum Ada Gambar!!!")