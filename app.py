import streamlit as st
from model import predict

st.set_page_config(page_title="Bone X-Ray Diagnosis", layout="centered")
st.title("🦴 Bone X-Ray Abnormality Detection")
st.write("Upload a bone X-ray image to predict whether it is **Normal** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    st.image("temp.jpg", caption="Uploaded X-ray", use_column_width=True)
    result = predict("temp.jpg")
    st.success(f"✅ Prediction: {result}")
