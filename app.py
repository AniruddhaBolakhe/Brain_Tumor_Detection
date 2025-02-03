import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Configure the generative AI model with your API key
genai.configure(api_key="AIzaSyB2G7Xyl8i74UQnAOyfP3Il9PU5OC72Alo")

# Define the tumor types
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Load model function
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.
    - Converts to BGR (OpenCV format).
    - Resizes to match model input dimensions (224x224).
    - Normalizes pixel values to range [0, 1].
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(model, image):
    """
    Predicts the likelihood of brain tumor presence and type.
    Returns:
    - predicted_class: Index of the highest confidence class.
    - confidence: Probability associated with the prediction.
    """
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    return predicted_class[0], confidence

def fetch_gemini_insights(tumor_type):
    """
    Fetch detailed insights on the tumor type from Gemini AI.
    """
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
model_options = {
    "VGGNet": "C:/Users/User/Desktop/Brain_Tumor/model.h5",
    "EfficientNet": "C:/Users/User/Desktop/Brain_Tumor/model_EfficientNetB0.h5",
    "Inception": "C:/Users/User/Desktop/Brain_Tumor/model_InceptionV3.h5",
    
}
selected_model_name = st.sidebar.selectbox("Select CNN Model", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# Upload multiple images
st.header("📋 Upload MRI Scans (All Views)")
uploaded_files = st.file_uploader(
    "Upload the Left, Right, Top, and Bottom views of the brain MRI scan (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 4:
    try:
        # Load the selected model
        model = load_model(selected_model_path)

        # Process and predict each image
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

        # Display combined results
        st.subheader("🩺 Diagnostic Report (Combined Views)")
        for view, diagnosis, confidence_percent in results:
            st.markdown(f"**View**: {view}  **Predicted Condition**: {diagnosis}  **Confidence Score**: {confidence_percent:.2f}%")

        # Fetch Gemini insights about the tumor
        tumor_type = results[0][1]  # Get the tumor type from the first view's prediction
        insights = fetch_gemini_insights(tumor_type)

        # Display detailed insights from Gemini
        st.subheader(f"💡 Detailed Information About {tumor_type}")
        st.markdown(insights)

        # Summary recommendation with balloons
        if all(diagnosis == "No Tumor" for _, diagnosis, _ in results):
            st.success("**Final Interpretation**: No evidence of a tumor detected across all uploaded views.")
            st.balloons()  # Trigger balloons animation for No Tumor
        else:
            st.error("**Final Interpretation**: Tumor detected in one or more views. Please consult a neurologist or radiologist for further evaluation.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
elif uploaded_files:
    st.warning("⚠️ Please upload exactly 4 images (Left, Right, Top, and Bottom views).")
else:
    st.info("Upload MRI scans to begin the analysis.")
