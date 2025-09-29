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
# (Prefer: put your key in st.secrets["GEMINI_API_KEY"])
genai.configure(api_key="AIzaSyC3-6CYA2z4sqtAVBAjdUKsYiANsi6zfqA")

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# Google Drive file IDs (unchanged)
model_drive_links = {
    "VGGNet": "1uS5vjUPWXJOpNREdzKkx_7AZWwqwOToY",
    "EfficientNet": "105GNzjRlc9z7AIQKbTFoY3GQBaNRkepb",
    "Inception": "15QeQquQ_-IoOmGy8ZLOaG64VPBgzqD76",
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
    # Common custom ops used by EfficientNet/TF Hub/etc.
    return {
        "swish": tf.nn.swish,
        "relu6": tf.nn.relu6,
        "tf": tf,  # sometimes models reference tf ops in Lambda layers
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

def get_model_input_hw(model) -> tuple[int, int]:
    """
    Returns (H, W) from model.input_shape. Handles shapes like:
    (None, H, W, C) or (H, W, C) or nested lists (for multi-input models).
    """
    shape = model.input_shape
    # If model has multiple inputs, take the first
    if isinstance(shape, list) or isinstance(shape, tuple) and isinstance(shape[0], (list, tuple)):
        shape = shape[0]

    # shape now expected like (None, H, W, C) or (H, W, C)
    if len(shape) == 4:
        _, H, W, C = shape
    elif len(shape) == 3:
        H, W, C = shape
    else:
        raise ValueError(f"Unexpected input shape: {shape}")

    if C != 3:
        raise ValueError(f"Model expects {C} channels; this app supports RGB (3 channels) only.")
    if H is None or W is None:
        # fallback to common sizes (rare)
        H, W = 224, 224
    return int(H), int(W)

def preprocess_image_for_model(image_pil: Image.Image, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Convert PIL->RGB numpy, resize to target (H, W), scale to [0,1], add batch dim.
    We keep simple 0..1 scaling since your VGG works that way; most custom-trained
    Inception/EfficientNet models will also be fine with this normalization.
    """
    image = image_pil.convert("RGB")
    np_img = np.array(image)
    H, W = target_hw
    np_img = cv2.resize(np_img, (W, H), interpolation=cv2.INTER_AREA)  # cv2 uses (W,H)
    np_img = np_img.astype("float32") / 255.0
    np_img = np.expand_dims(np_img, axis=0)  # (1, H, W, 3)
    return np_img

def predict_single(model, image_pil: Image.Image):
    target_hw = get_model_input_hw(model)
    x = preprocess_image_for_model(image_pil, target_hw)
    preds = model.predict(x, verbose=0)
    if preds.ndim == 1:
        # shape (num_classes,)
        pred_idx = int(np.argmax(preds))
        conf = float(np.max(preds))
    else:
        # shape (1, num_classes)
        pred_idx = int(np.argmax(preds, axis=1)[0])
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
This system uses trained CNN models to assist in identifying brain tumors from MRI scans.

Supported classes:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**
""")

st.sidebar.title("Navigation")
st.sidebar.info("Upload images and select the model for analysis.")

# Model selector & download
selected_model_name = st.sidebar.selectbox("Select CNN Model", list(model_drive_links.keys()))
selected_model_path = download_model(selected_model_name)

st.header("📤 Upload MRI Scans (All Views)")
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

            pred_idx, confidence = predict_single(model, image)
            diagnosis = CLASS_NAMES[pred_idx] if 0 <= pred_idx < len(CLASS_NAMES) else f"Class {pred_idx}"
            results.append((views[idx], diagnosis, confidence * 100.0))

        st.subheader("🩺 Diagnostic Report (Combined Views)")
        for view, diagnosis, conf_pct in results:
            st.markdown(f"**View**: {view} &nbsp;&nbsp; **Predicted Condition**: {diagnosis} &nbsp;&nbsp; **Confidence**: {conf_pct:.2f}%")

        # Provide educational insights for the first predicted class
        tumor_type = results[0][1]
        st.subheader(f"📚 Detailed Information About {tumor_type}")
        st.markdown(fetch_gemini_insights(tumor_type))

        # Final interpretation
        if all(diagnosis == "No Tumor" for _, diagnosis, _ in results):
            st.success("**Final Interpretation**: No evidence of a tumor detected across all uploaded views.")
            st.balloons()
        else:
            st.error("**Final Interpretation**: Tumor detected in one or more views. Please consult a neurologist/radiologist for further evaluation.")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
elif uploaded_files:
    st.warning("Please upload exactly 4 images (Left, Right, Top, and Bottom views).")
else:
    st.info("Upload MRI scans to begin the analysis.")

# Footer
st.markdown("---")
st.markdown("#### 👥 Project Contributors")
st.markdown("""
- [Aniruddha Bolakhe](https://www.linkedin.com/in/aniruddha-bolakhe-3b5090247/)
- [Nabhya Sharma](https://www.linkedin.com/in/nabhya-sharma-b0a374248/)
- [Pranav Karwa](https://www.linkedin.com/in/pranav-karwa-a91663249)
""")
