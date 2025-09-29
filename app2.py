import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import gdown
import os

# =========================
# CONFIG
# =========================
# Tip: prefer st.secrets["GEMINI_API_KEY"]
genai.configure(api_key="AIzaSyC3-6CYA2z4sqtAVBAjdUKsYiANsi6zfqA")

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Only VGGNet now
model_drive_links = {
    "VGGNet": "1uS5vjUPWXJOpNREdzKkx_7AZWwqwOToY",
}

os.makedirs("models", exist_ok=True)

# =========================
# HELPERS
# =========================
def download_model(model_name: str) -> str:
    file_id = model_drive_links[model_name]
    model_path = f"models/{model_name}.h5"
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name} model…")
        try:
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {model_name}: {e}")
            raise
        st.success(f"{model_name} downloaded successfully!")
    return model_path

def _get_custom_objects():
    # Common custom ops used in many TF/Keras exports
    return {
        "swish": tf.nn.swish,
        "relu6": tf.nn.relu6,
        "tf": tf,  # for Lambda layers referencing tf.*
    }

@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=_get_custom_objects(),
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        raise

def get_model_input_hw(model):
    """
    Returns (H, W) from model.inputs[0].shape, robust for single/multi-IO models.
    Expected NHWC. Falls back to 224x224 if unknown.
    """
    shape = model.inputs[0].shape  # TensorShape like (None, H, W, C)
    H = int(shape[1]) if shape[1] is not None else 224
    W = int(shape[2]) if shape[2] is not None else 224
    C = int(shape[3]) if shape[3] is not None else 3
    if C != 3:
        raise ValueError(f"Model expects {C} channels; this app supports RGB (3) only.")
    return H, W

def preprocess_image_for_model(image_pil: Image.Image, target_hw):
    """
    PIL -> RGB np, resize to (H, W), scale to [0,1], add batch dim.
    """
    image = image_pil.convert("RGB")
    np_img = np.array(image)
    H, W = target_hw
    np_img = cv2.resize(np_img, (W, H), interpolation=cv2.INTER_AREA)  # cv2: (W,H)
    np_img = np_img.astype("float32") / 255.0
    np_img = np.expand_dims(np_img, axis=0)  # (1, H, W, 3)
    return np_img

def predict_single(model, image_pil: Image.Image):
    target_hw = get_model_input_hw(model)
    x = preprocess_image_for_model(image_pil, target_hw)
    preds = model.predict(x, verbose=0)

    # Handle logits vs probabilities
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]
    if preds.ndim != 1:
        preds = np.squeeze(preds)

    # softmax if not already
    if not np.allclose(np.sum(preds), 1.0, atol=1e-3):
        preds = tf.nn.softmax(preds).numpy()

    pred_idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    return pred_idx, conf

def fetch_gemini_insights(tumor_type: str) -> str:
    prompt = (
        f"You are an experienced neuro-oncologist. "
        f"Explain {tumor_type} in simple, patient-friendly terms. "
        f"Cover typical symptoms, diagnostic workflow (MRI/biopsy if relevant), "
        f"standard treatments (surgery/radiation/chemotherapy, targeted/adjunct if applicable), "
        f"prognosis factors, and red-flag situations requiring urgent care."
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching insights: {str(e)}"

# =========================
# UI
# =========================
st.title("🧠 Brain Tumor Detection System")
st.markdown("""
Upload a single MRI image. The app will run your **VGGNet** model and provide a predicted class
along with patient-friendly information via Gemini 2.5 Flash.

**Classes**
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor
""")

st.sidebar.title("Model")
selected_model_name = st.sidebar.selectbox("Select CNN Model", ["VGGNet"])
selected_model_path = download_model(selected_model_name)

st.header("📤 Upload a Single MRI Image")
uploaded_file = st.file_uploader(
    "Upload one MRI image (JPG/JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        # Show image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_container_width=True)

        # Load model & predict
        model = load_model(selected_model_path)
        pred_idx, confidence = predict_single(model, image)

        diagnosis = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
        st.subheader("🩺 Prediction")
        st.markdown(f"**Predicted Condition:** {diagnosis}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")

        # Patient-friendly info
        st.subheader(f"📚 About {diagnosis}")
        st.markdown(fetch_gemini_insights(diagnosis))

        # Simple interpretation
        if diagnosis == "No Tumor":
            st.success("No evidence of a tumor detected in the uploaded image.")
        else:
            st.error("Tumor features detected in this image. Please consult a neurologist/radiologist for further evaluation.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("Upload an image to begin the analysis.")

# Footer
st.markdown("---")
st.markdown("#### 👥 Project Contributors")
st.markdown("""
- [Aniruddha Bolakhe](https://www.linkedin.com/in/aniruddha-bolakhe-3b5090247/)
- [Nabhya Sharma](https://www.linkedin.com/in/nabhya-sharma-b0a374248/)
- [Pranav Karwa](https://www.linkedin.com/in/pranav-karwa-a91663249)
""")
