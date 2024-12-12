import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown


os.makedirs('models', exist_ok=True)

# models
cnn_model_url = "https://drive.google.com/uc?id=1Ka9r40Z3bv3EzUuKgGVEeRy663Qme-ry"
vgg16_model_url = "https://drive.google.com/uc?id=1xQNCeJ44RKszqQ_PlWXnqK2qaf2kgQvL"

cnn_model_path = "models/cnn_model.h5"
vgg16_model_path = "models/vgg16_model.h5"

if not os.path.exists(cnn_model_path):
    gdown.download(cnn_model_url, cnn_model_path, quiet=False)
if not os.path.exists(vgg16_model_path):
    gdown.download(vgg16_model_url, vgg16_model_path, quiet=False)

cnn_model = load_model(cnn_model_path)
vgg16_model = load_model(vgg16_model_path)

def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(model, image):
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    probabilities = predictions[0]
    return predicted_class, probabilities

st.title("Fashion App")

st.sidebar.title("Select Model")
model_choice = st.sidebar.radio("Choose a model:", ("CNN Model", "VGG16 Model"))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    preprocessed_image = preprocess_image(image, target_size=(224, 224))

    model = cnn_model if model_choice == "CNN Model" else vgg16_model

    predicted_class, probabilities = predict(model, preprocessed_image)

    st.subheader("Prediction Results")
    st.write(f"Predicted Class: {predicted_class}")
    st.write("Class Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"Class {i}: {prob:.4f}")

    st.write("Note: Add loss/accuracy graphs if you have saved them.")