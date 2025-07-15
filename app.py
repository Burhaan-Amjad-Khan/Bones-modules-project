# app.py

import streamlit as st
from model import predict

st.title("🦴 Bone X-ray Fracture Classification")

uploaded_file = st.file_uploader("Upload an X-ray Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
    
    if st.button("Predict"):
        result = predict("temp_image.png")
        st.success(f"Prediction: {result}")
