import streamlit as st
import google.generativeai as genai
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import gdown
import os

# -----------------------------
# Configure the generative AI model with your API key (hardcoded)
# (Tip: use st.secrets or an env var for production)
# -----------------------------
genai.configure(api_key="AIzaSyC3-6CYA2z4sqtAVBAjdUKsYiANsi6zfqA")

# -----------------------------
# Define the tumor types
# -----------------------------
CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

# -----------------------------
# Google Drive file IDs
# -----------------------------
model_drive_links = {
    "VGGNet": "1uS5vjUPWXJOpNREdzKkx_7AZWwqwOToY",
    "EfficientNet": "105GNzjRlc9z7AIQKbTFoY3GQBaNRkepb",
    "Inception": "15QeQquQ_-IoOmGy8ZLOaG64VPBgzqD76",
}

# -----------------------------
# Per-model input sizes & preprocessing
# -----------------------------
MODEL_CONFIG = {
    "VGGNet":      {"size": (224, 224), "preprocess": "normalize_01"},
    "EfficientNet":{"size": (224, 224), "preprocess": "normalize_01"},  # change to "efficientnet" if that’s how you trained
    "Inception":   {"size": (299, 299), "preprocess": "inception"},     # inception commonly expects [-1,1]
}

# Optional keras.applications preprocessors (used if available & selected)
try:
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
    HAS_INCEPTION_PREP = True
except Exception:
    HAS_INCEPTION_PREP = False

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    HAS_EFF_PREP = True
except Exception:
    HAS_EFF_PREP = False

# -----------------------------
# Ensure model directory exists
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Function to download model
# -----------------------------
def download_model(model_name):
    file_id = model_drive_links[model_name]
    model_path = f"models/{model_name}.h5"
    if not os.path.exists(model_path):  # Avoid re-downloading
        st.info(f"Downloading {model_name} model... ")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        st.success(f"{model_name} downloaded successfully!")
    return model_path

# -----------------------------
# Robust Flatten shim & model loader (handles Flatten()([x]) legacy graphs)
# -----------------------------
def load_model(model_path):
    # Clear previous graphs so hot-reload doesn't keep stale classes
    tf.keras.backend.clear_session()

    class FlattenShim(tf.keras.layers.Layer):
        """
        Drop-in for saved graphs that did Flatten()([x]).
        Unwraps list/tuple inputs, then delegates to a real Flatten
        so static shape inference is preserved for downstream Dense layers.
        """
        def __init__(self, *args, **kwargs):
            kwargs.pop("data_format", None)  # ignore unknown legacy kwarg
            super().__init__(*args, **kwargs)
            self._flatten = tf.keras.layers.Flatten()

        def build(self, input_shape):
            s = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
            self._flatten.build(s)
            super().build(s)

        def call(self, inputs):
            x = inputs
            while isinstance(x, (list, tuple)) and len(x) > 0:
                x = x[0]
            return self._flatten(x)

        def compute_output_shape(self, input_shape):
            s = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
            return self._flatten.compute_output_shape(s)

        def get_config(self):
            return super().get_config()

    custom_objects = {
        "Flatten": FlattenShim,
        "keras.layers.core.flatten.Flatten": FlattenShim,  # fqcn mapping if present
    }

    try:
        return tf.keras.models.load_model(
            model_path, compile=False, custom_objects=custom_objects, safe_mode=False
        )
    except TypeError:
        # older TF/Keras without safe_mode
        return tf.keras.models.load_model(
            model_path, compile=False, custom_objects=custom_objects
        )

# -----------------------------
# Globals set after selection
# -----------------------------
CURRENT_SIZE = (224, 224)
CURRENT_PREP  = "normalize_01"

def preprocess_image(image):
    """
    image: numpy RGB array
    Resizes per model and applies simple preprocessing:
      - normalize_01: scale to [0,1]
      - inception: use keras inception preprocess if available else scale to [-1,1]
      - efficientnet: use keras efficientnet preprocess if available else [0,1]
    """
    img = cv2.resize(image, CURRENT_SIZE, interpolation=cv2.INTER_AREA).astype("float32")

    if CURRENT_PREP == "normalize_01":
        img = img / 255.0
    elif CURRENT_PREP == "inception":
        if HAS_INCEPTION_PREP:
            img = inception_preprocess(img)
        else:
            img = (img / 127.5) - 1.0  # fallback to [-1,1]
    elif CURRENT_PREP == "efficientnet":
        if HAS_EFF_PREP:
            img = efficientnet_preprocess(img)
        else:
            img = img / 255.0
    else:
        img = img / 255.0

    return np.expand_dims(img, axis=0)

def predict(model, image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction)
    return predicted_class[0], confidence

# -----------------------------
# Gemini insights (prefers gemini-2.5-flash with fallbacks)
# -----------------------------
def fetch_gemini_insights(tumor_type):
    prompt = (
        f"Provide concise, clinician-style information for {tumor_type} in adult patients. "
        "Cover: typical symptoms, initial evaluation, common treatments, expected prognosis, "
        "and red-flag signs requiring urgent care. Bullet points + short paragraphs. Not medical advice."
    )
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    last_err = ""
    for mid in models_to_try:
        try:
            m = genai.GenerativeModel(mid)
            resp = m.generate_content(prompt)
            return getattr(resp, "text", "No response text.")
        except Exception as e:
            last_err = str(e)
            continue
    return f"Error fetching insights: {last_err}"

# -----------------------------
# Streamlit UI
# -----------------------------
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

# Configure preprocessing based on selected model
mcfg = MODEL_CONFIG[selected_model_name]
CURRENT_SIZE = mcfg["size"]
CURRENT_PREP  = mcfg["preprocess"]

# Optional toggle to use keras EfficientNet preprocessing if you trained with it
if selected_model_name == "EfficientNet" and HAS_EFF_PREP:
    if st.sidebar.checkbox("Use keras EfficientNet preprocessing", value=False):
        CURRENT_PREP = "efficientnet"

# -----------------------------
# Upload multiple images
# -----------------------------
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
            # Always ensure RGB (avoids grayscale/RGBA issues)
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"{views[idx]} View", use_container_width=True)

            image_np = np.array(image)  # already RGB; DON'T swap to BGR
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

# -----------------------------
# Footer Section
# -----------------------------
st.markdown("---")
st.markdown("####  Project Contributors")
st.markdown("""
- [Aniruddha Bolakhe](https://www.linkedin.com/in/aniruddha-bolakhe-3b5090247/)
- [Nabhya Sharma](https://www.linkedin.com/in/nabhya-sharma-b0a374248/)
- [Pranav Karwa](https://www.linkedin.com/in/pranav-karwa-a91663249)
""")
