import streamlit as st
from PIL import Image
from predict import predict_image

st.title("Image Classification using CNN")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

if uploaded:

    img = Image.open(uploaded)

    st.image(img,width=250)

    result = predict_image(img)

    st.success(f"Prediction : {result}")
