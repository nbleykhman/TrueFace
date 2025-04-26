import io
import streamlit as st
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from model import get_model
from config import device, CHECKPOINT
import torch.nn.functional as F
import qrcode

# --- Streamlit setup ---
st.set_page_config(page_title="TrueFace", layout="centered")
st.title("TrueFace: Real vs. Fake Face Detector")
st.write("Upload a clear face image and TrueFace will tell you if it's real or AI-generated.")

# --- Load your classifier model ---
@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Initialize MTCNN for face detection ---
mtcnn = MTCNN(keep_all=False, device=device)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    # load & show original
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # detect face box
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        st.error("No face detected. Please upload a different image.")
    else:
        # take the first box
        x1, y1, x2, y2 = boxes[0]
        # add a small padding
        pad = 0.1 * max(x2 - x1, y2 - y1)
        left   = max(0, int(x1 - pad))
        top    = max(0, int(y1 - pad))
        right  = min(image.width,  int(x2 + pad))
        bottom = min(image.height, int(y2 + pad))

        face_img = image.crop((left, top, right, bottom))
        st.image(face_img, caption="Cropped Face", use_container_width=True)

        # preprocess for your classifier
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        tensor = preprocess(face_img).unsqueeze(0).to(device)

        # run inference
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

# --- QR Code for quick access ---
DEPLOY_URL = "https://face-classification.streamlit.app"
qr_img = qrcode.make(DEPLOY_URL)
buf = io.BytesIO()
qr_img.save(buf, format="PNG")
buf.seek(0)
st.image(buf, width=150, caption=DEPLOY_URL)


