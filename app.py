import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Lung Cancer Detection",
    page_icon="🫁",
    layout="wide"
)

# ---------------- THEME SWITCH ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme = st.sidebar.radio("Theme Mode", ["Light", "Dark"])
st.session_state.theme = theme


# ---------------- LIGHT MODE (PINKISH BROWN) ----------------
if theme == "Light":

    st.markdown("""
    <style>

    .stApp{
    background: linear-gradient(135deg,#f6c1cc,#e7a9b7);
    }

    header{
    background:#e7a9b7 !important;
    }

    section[data-testid="stSidebar"]{
    background:#8b4a5a !important;
    color:white;
    }

    [data-testid="stFileUploader"]{
    background:#f2b6c2;
    border-radius:15px;
    padding:15px;
    }

    .stButton>button{
    background:#a34863;
    color:white;
    border-radius:10px;
    border:none;
    padding:10px 20px;
    }

    .stButton>button:hover{
    background:#7c3449;
    }

    .card{
    background:#ffd6dd;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 10px 25px rgba(0,0,0,0.15);
    }

    .title{
    font-size:55px;
    font-weight:800;
    text-align:center;
    color:#7c1f3a;
    }

    .subtitle{
    text-align:center;
    font-size:20px;
    color:#4d2b33;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- DARK MODE ----------------
else:

    st.markdown("""
    <style>

    .stApp{
    background:#0e1117;
    color:white;
    }

    section[data-testid="stSidebar"]{
    background:#000;
    }

    [data-testid="stFileUploader"]{
    background:#1e1e1e;
    }

    .stButton>button{
    background:#222;
    color:white;
    border-radius:10px;
    }

    .title{
    font-size:55px;
    font-weight:800;
    text-align:center;
    color:#4fc3f7;
    }

    .subtitle{
    text-align:center;
    font-size:20px;
    color:#cfd8dc;
    }

    .card{
    background:#1c1f26;
    padding:25px;
    border-radius:20px;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🫁 AI Lung Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Early Detection Can Save Millions of Lives ❤️</div>', unsafe_allow_html=True)

st.write("Upload a **Lung CT Scan image** and allow AI to analyze potential cancer detection.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("About AI System")

st.sidebar.write("""
This AI system analyzes lung CT scan images using a deep learning model to detect signs of lung cancer.
""")

st.sidebar.markdown("---")

st.sidebar.subheader("Lung Cancer Awareness")

st.sidebar.write("""
Common Symptoms:

• Persistent cough  
• Chest pain  
• Shortness of breath  
• Fatigue  
""")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ai_model():
    return load_model("lung_cancer_model.h5")

model = load_ai_model()

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Lung Scan Image",
    type=["jpg","jpeg","png"]
)

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)
    return img

# ---------------- MAIN APP ----------------
if uploaded_file:

    image = Image.open(uploaded_file)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Lung Scan", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.button("Analyze Scan 🔍"):

            img = preprocess(image)

            prediction = model.predict(img)

            prob = prediction[0][0]

            if prob > 0.5:
                result = "Cancer Detected"
                confidence = prob*100
                st.error("⚠ Cancer Detected")
            else:
                result = "Non Cancerous"
                confidence = (1-prob)*100
                st.success("✅ Non Cancerous")

            st.progress(int(confidence))

            st.write(f"Confidence Score: **{confidence:.2f}%**")

            labels=["Cancer","Normal"]
            values=[prob,1-prob]

            fig,ax = plt.subplots()
            ax.bar(labels,values)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

            st.session_state.history.append({
                "Result":result,
                "Confidence":confidence
            })

            report = f"""
AI Lung Cancer Detection Report

Result : {result}
Confidence : {confidence:.2f}%
"""

            st.download_button(
                "Download Medical Report",
                report,
                file_name="lung_report.txt"
            )

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY TABLE ----------------
if st.session_state.history:

    st.markdown("### Scan History")

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("AI Healthcare System • Built with Deep Learning ")
