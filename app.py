import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("Digit Classifier - Deep Learning Model Deployment")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input

    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Preprocess image
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Show prediction
    st.success(f"Predicted Digit: **{predicted_class}**")
