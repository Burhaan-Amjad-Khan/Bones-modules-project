import os
os.environ["XDG_CONFIG_HOME"] = "."

import streamlit as st
from model import predict

st.title("Bone Abnormality Detection")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = predict("temp_image.png")
    st.image("temp_image.png", caption="Uploaded X-ray", use_column_width=True)
    st.write(f"Prediction: **{result}**")
