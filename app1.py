import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import gdown
import os

# Configure the generative AI model with your API key (hardcoded)
genai.configure(api_key="AIzaSyB2G7Xyl8i74UQnAOyfP3Il9PU5OC72Alo")

# Define the tumor types
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Google Drive file IDs
model_drive_links = {
    "VGGNet": "1uS5vjUPWXJOpNREdzKkx_7AZWwqwOToY",
    "EfficientNet": "105GNzjRlc9z7AIQKbTFoY3GQBaNRkepb",
    "Inception": "15QeQquQ_-IoOmGy8ZLOaG64VPBgzqD76",
}

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Function to download model
def download_model(model_name):
    file_id = model_drive_links[model_name]
    model_path = f"models/{model_name}.h5"
    
    if not os.path.exists(model_path):  # Avoid re-downloading
        st.info(f"Downloading {model_name} model... ")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        st.success(f"{model_name} downloaded successfully!")
    return model_path

# Load model function
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(model, image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    return predicted_class[0], confidence

def fetch_gemini_insights(tumor_type):
    prompt = f"Please provide detailed information about {tumor_type}. Include symptoms, treatment options, and prognosis, give the information like a doctor."
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights: {str(e)}"

# Streamlit UI
st.title(" Brain Tumor Detection System")
st.markdown("""
This system leverages trained neural networks to assist in identifying brain tumors through MRI scans. 
It supports the following tumor types:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor** (normal brain scan)
""")

st.sidebar.title("Navigation")
st.sidebar.info("Upload images and select the model for analysis.")

# Dropdown for selecting the model
selected_model_name = st.sidebar.selectbox("Select CNN Model", list(model_drive_links.keys()))
selected_model_path = download_model(selected_model_name)

# Upload multiple images
st.header(" Upload MRI Scans (All Views)")
uploaded_files = st.file_uploader(
    "Upload the Left, Right, Top, and Bottom views of the brain MRI scan (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 4:
    try:
        model = load_model(selected_model_path)

        results = []
        views = ["Left", "Right", "Top", "Bottom"]
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            st.image(image, caption=f"{views[idx]} View", use_container_width=True)

            image_np = np.array(image)
            predicted_class, confidence = predict(model, image_np)

            diagnosis = CLASS_NAMES[predicted_class]
            confidence_percent = confidence * 100
            results.append((views[idx], diagnosis, confidence_percent))

        st.subheader("🩺 Diagnostic Report (Combined Views)")
        for view, diagnosis, confidence_percent in results:
            st.markdown(f"**View**: {view}  **Predicted Condition**: {diagnosis}  **Confidence Score**: {confidence_percent:.2f}%")

        tumor_type = results[0][1]
        insights = fetch_gemini_insights(tumor_type)

        st.subheader(f" Detailed Information About {tumor_type}")
        st.markdown(insights)

        if all(diagnosis == "No Tumor" for _, diagnosis, _ in results):
            st.success("**Final Interpretation**: No evidence of a tumor detected across all uploaded views.")
            st.balloons()
        else:
            st.error("**Final Interpretation**: Tumor detected in one or more views. Please consult a neurologist or radiologist for further evaluation.")
    except Exception as e:
        st.error(f" An error occurred: {e}")
elif uploaded_files:
    st.warning(" Please upload exactly 4 images (Left, Right, Top, and Bottom views).")
else:
    st.info("Upload MRI scans to begin the analysis.")

# Footer Section
st.markdown("---")
st.markdown("####  Project Contributors")
st.markdown("""
- [Aniruddha Bolakhe](https://www.linkedin.com/in/aniruddha-bolakhe-3b5090247/)
- [Nabhya Sharma](https://www.linkedin.com/in/nabhya-sharma-b0a374248/)
- [Pranav Karwa](https://www.linkedin.com/in/pranav-karwa-a91663249)
""")
