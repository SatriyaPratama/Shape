import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

IMG_WIDTH = 200
IMG_HEIGHT = 200

# Verify if model loads correctly
try:
    model = load_model('shape_detector_model2.keras')
    st.write("Model loaded successfully.")
except Exception as e:
    st.write("Error loading model:", e)

def preprocess_image(image):
    # Resize image to the size your model expects
    image = image.convert('RGB')
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    
    # Debug: Show raw predictions
    st.write("Raw Model Prediction:", prediction)
    
    shape = np.argmax(prediction, axis=1)[0]
    return shape

# Streamlit UI
st.title("Shape Predictor")
st.write("Upload an image to predict its shape")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    shape = predict(image)
    shape_dict = {0: 'Circle', 1: 'Square', 2: 'Triangle', 3: 'Lainnya'}

    st.write(f"Predicted Shape: {shape_dict[shape]}")
