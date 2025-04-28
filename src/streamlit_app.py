import os
import io
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from model import get_model
from config import device, FINETUNE_EMA_CHECKPOINT, FINETUNE_RESOLUTION
import qrcode

# --- Streamlit layout ---
st.set_page_config(page_title="TrueFace", layout="centered")
st.title("TrueFace: Real vs. Fake Face Detector")
st.write("Upload a clear face image and TrueFace will tell you if it's real or AI-generated.")

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@st.cache_resource
def load_threshold():
    base = os.path.dirname(__file__)
    thr_path = os.path.join(base, "..", "EMApipeline", "evaluation_results", "combined_threshold.txt")
    try:
        return float(open(thr_path, "r").read().strip())
    except Exception as e:
        st.error(f"Could not load threshold from {thr_path}: {e}")
        return 0.5

threshold = load_threshold()

# --- Load classifier model ---
@st.cache_resource
def load_model():
    model = get_model()
    ckpt   = torch.load(FINETUNE_EMA_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- MTCNN for face detection ---
mtcnn = MTCNN(keep_all=False, device=device)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # detect face
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        st.error("No face detected. Please upload a different image.")
    else:
        x1, y1, x2, y2 = boxes[0]
        pad = 0.1 * max(x2-x1, y2-y1)
        left   = max(0, int(x1 - pad))
        top    = max(0, int(y1 - pad))
        right  = min(image.width,  int(x2 + pad))
        bottom = min(image.height, int(y2 + pad))
        face_img = image.crop((left, top, right, bottom))
        st.image(face_img, caption="Cropped Face", use_container_width=True)

        # preprocess
        preprocess = transforms.Compose([
            transforms.Resize(FINETUNE_RESOLUTION),
            transforms.CenterCrop(FINETUNE_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        tensor = preprocess(face_img).unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            logits    = model(tensor)
            probs     = F.softmax(logits, dim=1)[0]
            real_prob = probs[0].item()
            fake_prob = probs[1].item()

        # apply combined threshold
        if fake_prob > threshold:
            label = "FAKE ❌"
            confidence = fake_prob * 100
        else:
            label = "REAL ✅"
            confidence = real_prob * 100

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.1f}%  (threshold={threshold:.3f})")

DEPLOY_URL = "https://trueface-classifier.streamlit.app/"
qr_img = qrcode.make(DEPLOY_URL)
buf = io.BytesIO()
qr_img.save(buf, format="PNG")
buf.seek(0)
st.image(buf, width=150, caption=DEPLOY_URL)



