import io
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import get_model
from config import device, CHECKPOINT
import torch.nn.functional as F
import pandas as pd
import qrcode

from train import IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(page_title="UNC AI Face Classifier", layout="wide")

@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Inject UNC-styled CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;600&display=swap');

html, body {{
    font-family: 'Outfit', sans-serif;
    background-image: url('https://upload.wikimedia.org/wikipedia/en/2/2f/Rameses_UNC_Mascot.png');
    background-repeat: no-repeat;
    background-position: bottom right;
    background-size: 160px;
    background-attachment: fixed;
}}

h1, h2, h3 {{
    color: #4B9CD3;
    letter-spacing: 0.03em;
}}

h1::before {{ content: "🎓 "; }}
h2::before {{ content: "🏛 "; }}

.upload-wrapper .stFileUploader {{
    border: 2px dashed #4B9CD3;
    border-radius: 14px;
    padding: 2rem;
    margin-top: -2rem;
    box-shadow: 0 0 12px rgba(75, 156, 211, 0.3);
}}

.info-card {{
    background: rgba(75, 156, 211, 0.12);
    border-left: 5px solid #4B9CD3;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 2rem;
    font-size: 1.05rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}}

.glass {{
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.1);
    padding: 1.5em;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    transition: transform 0.3s ease;
}}

.glass:hover {{
    transform: scale(1.02);
}}

.confidence {{
    width: 100%;
    background-color: rgba(255,255,255,0.2);
    border-radius: 14px;
    margin-top: 0.5rem;
    height: 20px;
    overflow: hidden;
}}

.bar {{
    height: 20px;
    color: white;
    font-weight: bold;
    text-align: center;
    animation: fill 1s ease forwards;
    line-height: 20px;
}}

@keyframes fill {{
    from {{ width: 0%; }}
    to   {{ width: var(--width); }}
}}

footer {{
    text-align: center;
    margin-top: 4rem;
    font-size: 0.9rem;
    opacity: 0.9;
    border-top: 1px dashed #4B9CD3;
    padding-top: 1rem;
}}
</style>
""", unsafe_allow_html=True)

# Deployment URL
DEPLOY_URL = "https://deepfake560.streamlit.app/"

# Header
st.markdown("## 🧠 UNC AI Fake vs. Real Face Detector")

# Info Card
st.markdown("""
<div class="info-card">
    <strong>📘 Instructions:</strong><br>
    • Upload one or more face images.<br>
    • The model will classify each image as REAL or FAKE.<br>
    • Animated confidence bars will show certainty.<br>
    • You can export all results to CSV at the bottom.
</div>
""", unsafe_allow_html=True)

# Upload Box
st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("📥", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
results = []
if uploaded_files:
    cols = st.columns(3)
    for idx, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)
            confidence_pct = confidence.item() * 100
            label = "REAL ✅" if prediction.item() == 0 else "FAKE ❌"
            color = "#4CAF50" if prediction.item() == 0 else "#e53935"

            results.append({
                "Filename": file.name,
                "Prediction": label,
                "Confidence (%)": round(confidence_pct, 2)
            })

        with cols[idx % 3]:
            st.image(image, use_container_width=True)
            st.markdown(f"""
                <div class="glass">
                    <strong>{label}</strong>
                    <div class="confidence">
                        <div class="bar" style="background:{color}; width:{confidence_pct:.2f}%; --width:{confidence_pct:.2f}%;">
                            {confidence_pct:.2f}%
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# CSV Export
if results:
    df = pd.DataFrame(results)
    st.markdown("### 📄 Download Prediction Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# Footer
st.markdown("""
<footer>
    Built with 💙 by UNC Innovators · AI Capstone 2025
</footer>
""", unsafe_allow_html=True)

# QR Code for deployment URL
st.markdown("---")
st.write("### Scan to open this app:")
qr_img = qrcode.make(DEPLOY_URL)
buf = io.BytesIO()
qr_img.save(buf, format="PNG")
buf.seek(0)
st.image(buf, width=200, caption=DEPLOY_URL)
