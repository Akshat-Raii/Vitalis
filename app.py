# app.py
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import requests
import base64
import json
from dotenv import load_dotenv
import PyPDF2

# Set page config
st.set_page_config(page_title="Vitalis", layout="wide", page_icon="./image/health-report.png")

# Load environment variables
load_dotenv()

# --- Gemini API Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("🔴 GEMINI_API_KEY not set in .env")
    st.stop()

MODEL_NAME = "gemini-2.5-flash"        # good default, text + vision
# or: MODEL_NAME = "gemini-flash-latest"
# or: MODEL_NAME = "gemini-2.0-flash"

GEMINI_API_URL_TEXT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
)
GEMINI_API_URL_VISION = GEMINI_API_URL_TEXT  # same model for multimodal

# --- Helper Functions ---
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_base64_from_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_gemini_api(prompt, images=None):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": []}]}
    if images:
        payload["contents"][0]["parts"].append({"text": prompt})
        for img in images:
            payload["contents"][0]["parts"].append(
                {"inline_data": {"mime_type": "image/jpeg", "data": img}}
            )
        api_url = GEMINI_API_URL_VISION
    else:
        payload["contents"][0]["parts"].append({"text": prompt})
        api_url = GEMINI_API_URL_TEXT

    try:
        r = requests.post(api_url, headers=headers, data=json.dumps(payload))
        r.raise_for_status()
        result = r.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error: {e}"

# --- Disease-specific Prompts ---
def get_disease_specific_prompt(disease):
    disclaimer = "⚠️ Disclaimer: This is not medical advice. Please consult a doctor."
    prompts = {
        "Diabetes": f"Explain the diabetes test result simply. Suggest diet & lifestyle tips. {disclaimer}",
        "Heart Disease": f"Explain the heart disease test result simply. Suggest heart-healthy habits. {disclaimer}",
        "Parkinsons": f"Explain the Parkinson’s test result simply. Suggest exercise & lifestyle tips. {disclaimer}",
        "Brain Tumor": f"Analyze this brain MRI + mask. Explain simply & suggest brain health tips. {disclaimer}",
        "Breast Ultrasound": f"Analyze this breast ultrasound + mask. Explain simply & suggest breast health tips. {disclaimer}",
    }
    return prompts.get(disease, disclaimer)

# --- Load ML Models ---
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f"{working_dir}/files/xgboost_diabetes_model.sav", "rb"))
diabetes_scaler = pickle.load(open(f"{working_dir}/files/diabetes_scaler.sav", "rb"))

heart_model = pickle.load(open(f"{working_dir}/files/xgboost_heart_model.sav", "rb"))
heart_scaler = pickle.load(open(f"{working_dir}/files/heart_scaler.sav", "rb"))

parkinsons_model = pickle.load(open(f"{working_dir}/files/xgboost_parkinsons_model.sav", "rb"))
parkinsons_scaler = pickle.load(open(f"{working_dir}/files/parkinsons_scaler.sav", "rb"))

# --- Load Imaging Models ---
@st.cache_resource
def load_brain_tumor_model():
    with CustomObjectScope({"dice_coef": lambda y_true, y_pred: 0, "dice_loss": lambda y_true, y_pred: 0}):
        return tf.keras.models.load_model("files/model.h5") if os.path.exists("files/model.h5") else None

brain_model = load_brain_tumor_model()

@st.cache_resource
def load_breast_ultrasound_model():
    return tf.keras.models.load_model("files/my_model.h5") if os.path.exists("files/my_model.h5") else None

breast_model = load_breast_ultrasound_model()

# --- Sidebar ---
with st.sidebar:
    selected = option_menu(
        "Vitalis",
        ["🏠 Home",
         "🤖 Dr.ChatWell",
         "🩸 Diabetes Prediction",
         "❤️ Heart Disease Prediction",
         "🧍 Parkinsons Prediction",
         "🧠 Brain Tumor Segmentation",
         "🩻 Breast Ultrasound Segmentation",
         "ℹ️ About"],
        default_index=0,
    )

# --- Home ---
if selected == "🏠 Home":
    st.title("🚑 Vitalis")
    st.markdown("Early detection + AI insights for better health 💡")
    st.image("image/cover.gif")

# --- AI Health Assistant ---
if selected == "🤖 Dr.ChatWell":
    st.title("🤖 Dr.ChatWell")
    st.write("Chat with AI or upload your **full body report (PDF/Image)**")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for a, m in st.session_state.chat_history:
        with st.chat_message(a):
            st.markdown(m)

    report = st.file_uploader("📄 Upload Report", type=["pdf", "jpg", "jpeg", "png"])
    if report:
        with st.spinner("Analyzing..."):
            if report.type == "application/pdf":
                reader = PyPDF2.PdfReader(report)
                text = "".join([(p.extract_text() or "") for p in reader.pages])
                prompt = f"Analyze this health report:\n{text}\nSummarize, suggest lifestyle, disclaimer."
                resp = call_gemini_api(prompt)
            else:
                create_dir("uploaded_reports")
                path = os.path.join("uploaded_reports", report.name)
                with open(path, "wb") as f:
                    f.write(report.getbuffer())
                img_b64 = get_base64_from_image(path)
                resp = call_gemini_api(
                    "Analyze this health report image. Summarize + tips + disclaimer.",
                    images=[img_b64],
                )
            st.session_state.chat_history.append(("assistant", resp))
            with st.chat_message("assistant"):
                st.markdown(resp)

    if q := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.spinner("Thinking..."):
            resp = call_gemini_api(
                f"User asked: {q}\nGive safe health advice, no diagnosis, add disclaimer."
            )
        st.session_state.chat_history.append(("assistant", resp))
        with st.chat_message("assistant"):
            st.markdown(resp)

# --- Diabetes Prediction ---
if selected == "🩸 Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose")
    with col3:
        BloodPressure = st.text_input("Blood Pressure")
    with col1:
        SkinThickness = st.text_input("Skin Thickness")
    with col2:
        Insulin = st.text_input("Insulin")
    with col3:
        BMI = st.text_input("BMI")
    with col1:
        DPF = st.text_input("DPF")
    with col2:
        Age = st.text_input("Age")

    if st.button("Get Result"):
        try:
            vals = [float(x) for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
            scaled = diabetes_scaler.transform([vals])
            pred = diabetes_model.predict(scaled)[0]
            res = "Diabetic" if pred == 1 else "Not Diabetic"
            st.success(res)
            st.markdown(call_gemini_api(get_disease_specific_prompt("Diabetes")))
        except Exception as e:
            st.error(e)

# --- Heart Disease Prediction ---
if selected == "❤️ Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain")
    with col1:
        trestbps = st.text_input("Rest BP")
    with col2:
        chol = st.text_input("Cholesterol")
    with col3:
        fbs = st.text_input("FBS > 120")
    with col1:
        restecg = st.text_input("Rest ECG")
    with col2:
        thalach = st.text_input("Max HR")
    with col3:
        exang = st.text_input("Exercise Angina")
    with col1:
        oldpeak = st.text_input("Oldpeak")
    with col2:
        slope = st.text_input("Slope")
    with col3:
        ca = st.text_input("Vessels")
    with col1:
        thal = st.text_input("Thal")

    if st.button("Get Result"):
        try:
            vals = [float(x) for x in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
            scaled = heart_scaler.transform([vals])
            pred = heart_model.predict(scaled)[0]
            res = "Heart Disease" if pred == 1 else "No Heart Disease"
            st.success(res)
            st.markdown(call_gemini_api(get_disease_specific_prompt("Heart Disease")))
        except Exception as e:
            st.error(e)

# --- Parkinsons Prediction ---
if selected == "🧍 Parkinsons Prediction":
    st.title("🧠 Parkinson's Prediction")
    feats = [
        "fo", "fhi", "flo", "Jitter(%)", "Jitter(Abs)", "RAP", "PPQ", "DDP",
        "Shimmer", "Shimmer(dB)", "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    vals = []
    cols = st.columns(5)
    for i, f in enumerate(feats):
        with cols[i % 5]:
            vals.append(st.text_input(f))
    if st.button("Get Result"):
        try:
            vals = [float(x) for x in vals]
            scaled = parkinsons_scaler.transform([vals])
            pred = parkinsons_model.predict(scaled)[0]
            res = "Parkinson's Disease" if pred == 1 else "No Parkinson's"
            st.success(res)
            st.markdown(call_gemini_api(get_disease_specific_prompt("Parkinsons")))
        except Exception as e:
            st.error(e)

# --- Brain Tumor Segmentation ---
if selected == "🧠 Brain Tumor Segmentation":
    st.title("🧠 Brain Tumor Segmentation")
    if not brain_model:
        st.warning("Model not found")
    else:
        img_file = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])
        if img_file and st.button("Segment & Analyze"):
            create_dir("uploaded_images")
            path = os.path.join("uploaded_images", img_file.name)
            with open(path, "wb") as f:
                f.write(img_file.getbuffer())
            img = cv2.imread(path)  # BGR
            # use float32 for model input
            resized = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
            inp = np.expand_dims(resized, 0)
            pred = brain_model.predict(inp)[0]
            mask = (pred >= 0.5).astype(np.uint8) * 255
            mask = np.squeeze(mask)

            create_dir("generated_masks")
            mask_path = os.path.join("generated_masks", "mask_" + img_file.name)
            cv2.imwrite(mask_path, mask)

            st.image(path, caption="MRI", width=400)
            st.image(mask_path, caption="Predicted Mask", width=400)

            img_b64 = get_base64_from_image(path)
            mask_b64 = get_base64_from_image(mask_path)
            st.markdown(
                call_gemini_api(get_disease_specific_prompt("Brain Tumor"), images=[img_b64, mask_b64])
            )

# --- Breast Ultrasound Segmentation ---
if selected == "🩻 Breast Ultrasound Segmentation":
    st.title("🩻 Breast Ultrasound Segmentation")
    if not breast_model:
        st.warning("Model not found")
    else:
        img_file = st.file_uploader("Upload Ultrasound", type=["jpg", "jpeg", "png"])
        if img_file and st.button("Segment & Analyze"):
            create_dir("uploaded_ultrasound")
            path = os.path.join("uploaded_ultrasound", img_file.name)
            with open(path, "wb") as f:
                f.write(img_file.getbuffer())

            # Read as grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            # float32 [0,1] for model input
            resized = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0
            inp = np.expand_dims(np.expand_dims(resized, -1), 0)
            pred = breast_model.predict(inp)[0]
            mask = (pred >= 0.5).astype(np.uint8) * 255
            mask = np.squeeze(mask)

            create_dir("ultrasound_masks")
            mask_path = os.path.join("ultrasound_masks", "mask_" + img_file.name)
            cv2.imwrite(mask_path, mask)

            st.image(path, caption="Ultrasound", width=400)
            st.image(mask_path, caption="Predicted Mask", width=400)

            # ---- FIXED: convert dtype before cvtColor/overlay ----
            overlay_u8 = np.clip(resized * 255.0, 0, 255).astype(np.uint8)  # (H, W) uint8
            overlay_bgr = cv2.cvtColor(overlay_u8, cv2.COLOR_GRAY2BGR)      # BGR
            heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)                 # BGR
            blend_bgr = cv2.addWeighted(overlay_bgr, 0.7, heat, 0.3, 0.0)
            blend_rgb = cv2.cvtColor(blend_bgr, cv2.COLOR_BGR2RGB)
            st.image(blend_rgb, caption="Overlay", width=400)

            img_b64 = get_base64_from_image(path)
            mask_b64 = get_base64_from_image(mask_path)
            st.markdown(
                call_gemini_api(get_disease_specific_prompt("Breast Ultrasound"), images=[img_b64, mask_b64])
            )

# --- About ---
if selected == "ℹ️ About":
    st.title("🌟 About This Project")
    st.markdown(
        """
    A unified AI-powered healthcare assistant:  
    - 🩸 Diabetes | ❤️ Heart | 🧠 Parkinson’s predictions  
    - 🖼️ Brain Tumor & Breast Lesion Segmentation  
    - 🤖 Dr. ChatWell Health Assistant (chat + report uploads)  

    ⚠️ This is **not a replacement for doctors**. Always seek medical advice from professionals.
    """
    )
