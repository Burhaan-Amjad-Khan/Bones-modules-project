import streamlit as st
from model import predict

st.title("Bone Fracture Classification Module 🦴")
st.write("Upload an X-ray image to check if it is **Normal** or **Abnormal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)
    
    result = predict("temp.jpg")
    st.markdown(f"## Prediction: `{result}`")
