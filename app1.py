import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import gdown
import os

genai.configure(api_key="AIzaSyB2G7Xyl8i74UQnAOyfP3Il9PU5OC72Alo")


CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]
model_drive_links = {
    "VGGNet": "1uS5vjUPWXJOpNREdzKkx_7AZWwqwOToY",
    "EfficientNet": "105GNzjRlc9z7AIQKbTFoY3GQBaNRkepb",
    "Inception": "15QeQquQ_-IoOmGy8ZLOaG64VPBgzqD76",
}
os.makedirs("models", exist_ok=True)

def download_model(model_name):
    file_id = model_drive_links[model_name]
    model_path = f"models/{model_name}.h5"
    
    if not os.path.exists(model_path):  # Avoid re-downloading
        st.info(f"Downloading {model_name} model... ⏳")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        st.success(f"{model_name} downloaded successfully! ✅")
    return model_path
    
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
    prompt = f"""
    You are an experienced neurologist and brain tumor specialist. A patient has been diagnosed with {tumor_type}.  
    Explain the condition in a **simple and empathetic** manner, helping the patient understand:  
    - **What this tumor is** and how it affects the brain.  
    - **Symptoms they may experience** and why they occur.  
    - **Possible causes or risk factors.**  
    - **Treatment options**, including medications, surgery, or radiation, and their pros/cons.  
    - **Next steps**, like seeking a neurologist, further tests, or lifestyle changes.  
      
    Avoid complex medical jargon; explain in a way a patient with no medical background can understand.  
    """

    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights: {str(e)}"

# Streamlit UI
st.title(" Brain Tumor Detection ")
st.markdown("""
system trained on neural networks to assist in identifying brain tumors through MRI scans. 
Supports the  tumor types:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
""")

st.sidebar.title("Navigation")
st.sidebar.info("Upload images and select the model for analysis.")

# Dropdown for selecting the model
selected_model_name = st.sidebar.selectbox("Select CNN Model", list(model_drive_links.keys()))
selected_model_path = download_model(selected_model_name)

# Upload multiple images
st.header("📋 Upload MRI Scans (All Views)")
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

        st.subheader(f"💡 Detailed Information About {tumor_type}")
        st.markdown(insights)

        if all(diagnosis == "No Tumor" for _, diagnosis, _ in results):
            st.success("**Final Interpretation**: No evidence of a tumor detected across all uploaded views.")
            st.balloons()
        else:
            st.error("**Final Interpretation**: Tumor detected in one or more views. Please consult a neurologist or radiologist for further evaluation.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
elif uploaded_files:
    st.warning("⚠️ Please upload exactly 4 images (Left, Right, Top, and Bottom views).")
else:
    st.info("Upload MRI scans to begin the analysis.")
