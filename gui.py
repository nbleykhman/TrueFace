import io
import streamlit as st
import torch
from PIL import Image
import numpy as np
import face_recognition
from torchvision import transforms
from model import get_model
from config import device, CHECKPOINT
import torch.nn.functional as F
import qrcode

# Simple Streamlit GUI for TrueFace
st.set_page_config(page_title="TrueFace", layout="centered")

@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("TrueFace: Real vs. Fake Face Detector")
st.write("Upload a clear face image and TrueFace will tell you if it's real or AI-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Detect faces
    faces = face_recognition.face_locations(img_array)
    if not faces:
        st.error("No face detected. Please upload a different image.")
    else:
        # Use first face
        top, right, bottom, left = faces[0]
        pad = int(0.1 * max(right-left, bottom-top))
        h, w, _ = img_array.shape
        left = max(0, left-pad)
        top = max(0, top-pad)
        right = min(w, right+pad)
        bottom = min(h, bottom+pad)
        face_img = image.crop((left, top, right, bottom))

        st.image(face_img, caption="Cropped Face", use_container_width=True)

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = preprocess(face_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(tensor)
            probs = F.softmax(out, dim=1)[0]
            real_prob = probs[0].item()
            fake_prob = probs[1].item()
            label = "REAL ✅" if real_prob > fake_prob else "FAKE ❌"
            confidence = max(real_prob, fake_prob) * 100

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.1f}%")

st.markdown("---")

# QR Code for quick access
DEPLOY_URL = "https://face-classification.streamlit.app"
qr_img = qrcode.make(DEPLOY_URL)
buf = io.BytesIO()
qr_img.save(buf, format="PNG")
buf.seek(0)
st.image(buf, width=150, caption=DEPLOY_URL)

