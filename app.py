import streamlit as st
from model import predict

st.title("Bone X-Ray Analysis")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file)
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    result = predict("temp_image.png")
    st.success(f"Prediction: {result}")
