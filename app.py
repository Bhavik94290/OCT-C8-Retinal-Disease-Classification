import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms
from src.models import get_model
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pandas as pd
import html

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="OCT-C8 | Retinal Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASSES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]
DEVICE = torch.device("cpu")
MODEL_PATH = Path("models/densenet121_final.pt")
DATASET_PATH = Path("dataset/train")
IMAGES_PER_CLASS = 20

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =====================================================
# 2. PROFESSIONAL UI STYLING
# =====================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #10233f;
        --muted: #64748b;
        --teal: #0f9d91;
        --line: #e5ebf3;
        --panel: #ffffff;
        --soft: #f5f8fc;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(15,157,145,.07), transparent 27%),
            linear-gradient(180deg, #f8fafc 0%, #f3f6fb 100%);
        color: #172b4d;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #10233f;
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] * { color: #edf5ff !important; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stFileUploader"] section {
        background: #ffffff;
        border: 1.5px dashed #b9c9da;
        border-radius: 16px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #0f9d91;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 0;
        background: linear-gradient(135deg, #0f9d91, #087f8c);
        color: white;
        font-weight: 700;
        padding: .68rem 1.2rem;
        min-height: 2.7rem;
        box-shadow: 0 5px 14px rgba(15,157,145,.18);
        transition: all .2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0b897f, #076c79);
        border: 0;
        color: white;
        transform: translateY(-1px);
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(16,35,63,.035);
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stMetricValue"] {
        color: #10233f;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
    }
    .hero {
        background: linear-gradient(120deg, #10233f 0%, #173d5b 65%, #0f766e 100%);
        color: white;
        border-radius: 22px;
        padding: 1.7rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 32px rgba(16,35,63,.12);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "";
        position: absolute;
        right: -45px;
        top: -85px;
        width: 260px;
        height: 260px;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        color: #9fe8df;
        font-weight: 800;
        margin-bottom: .55rem;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.7rem, 3vw, 2.45rem);
        line-height: 1.16;
        color: white;
        margin: 0 0 .6rem 0;
        font-weight: 800;
    }
    .hero p { color: #d9e8f5; max-width: 760px; margin: 0; font-size: 1rem; }
    .hero-tag {
        display: inline-block;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,.22);
        background: rgba(255,255,255,.09);
        border-radius: 999px;
        padding: .35rem .75rem;
        font-size: .78rem;
        color: #effffb;
    }
    .section-heading {
        font-family: 'Manrope', sans-serif;
        color: #10233f;
        font-size: 1.25rem;
        font-weight: 800;
        margin: .25rem 0 .15rem 0;
    }
    .section-subtitle {
        color: #64748b;
        font-size: .9rem;
        margin-bottom: 1rem;
    }
    .panel {
        background: #ffffff;
        border: 1px solid #e5ebf3;
        border-radius: 17px;
        padding: 1.15rem;
        box-shadow: 0 7px 24px rgba(16,35,63,.035);
        height: 100%;
    }
    .panel-label {
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #64748b;
        font-weight: 800;
        margin-bottom: .5rem;
    }
    .prediction-class {
        color: #0b766e;
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        line-height: 1.2;
        font-weight: 800;
        margin: .25rem 0 .65rem 0;
    }
    .status-pill {
        display: inline-block;
        background: #e7f8f3;
        color: #087f70;
        border: 1px solid #b9eee1;
        border-radius: 999px;
        padding: .3rem .65rem;
        font-size: .76rem;
        font-weight: 700;
    }
    .note {
        background: #fff9e9;
        border: 1px solid #f6e6b3;
        border-radius: 12px;
        color: #775a13;
        padding: .85rem 1rem;
        font-size: .86rem;
        margin: .8rem 0 1.15rem 0;
    }
    .similarity-card {
        background: white;
        border: 1px solid #e5ebf3;
        border-radius: 14px;
        padding: .65rem;
        box-shadow: 0 5px 18px rgba(16,35,63,.04);
        height: 100%;
    }
    .rank {
        color: #0f9d91;
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: .45rem 0 .2rem 0;
    }
    .sim-score { color: #10233f; font-weight: 800; font-size: 1rem; }
    .class-tag {
        display: inline-block;
        background: #eef4fa;
        color: #405875;
        border-radius: 7px;
        padding: .22rem .48rem;
        margin-top: .45rem;
        font-size: .72rem;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #8290a3;
        font-size: .78rem;
        padding: 1.5rem 0 .4rem 0;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e5ebf3;
        border-radius: 12px;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# 3. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 👁️ OCT-C8")
    st.caption("RETINAL IMAGE RESEARCH")
    st.divider()
    st.markdown("### Analysis workspace")
    st.markdown("- **Model:** DenseNet121")
    st.markdown("- **Task:** 8-class classification")
    st.markdown("- **Runtime:** CPU")
    st.divider()
    st.markdown("### Supported classes")
    st.write("AMD · CNV · CSR · DME")
    st.write("DR · DRUSEN · MH · NORMAL")
    st.divider()
    st.caption("Research prototype • Not for clinical diagnosis")

# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    model = get_model("densenet121")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# =====================================================
# 5. FEATURE EXTRACTION
# =====================================================
def extract_features(model, tensor):
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        features = model.model.features(tensor)
        features = F.relu(features)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, start_dim=1)
        features = F.normalize(features, p=2, dim=1)
    return features.detach()


# =====================================================
# 6. REFERENCE FEATURE BANK
# =====================================================
@st.cache_data(show_spinner=False)
def build_feature_bank():
    model = load_model()
    feature_list = []
    image_paths = []

    if not DATASET_PATH.exists():
        return np.empty((0, 1024)), []

    class_directories = sorted(
        p for p in DATASET_PATH.iterdir() if p.is_dir()
    )

    for class_dir in class_directories:
        image_files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )[:IMAGES_PER_CLASS]

        for path in image_files:
            try:
                with Image.open(path) as img:
                    reference_image = img.convert("RGB")
                tensor = TRANSFORM(reference_image).unsqueeze(0)
                feature = extract_features(model, tensor)
                feature_list.append(feature.cpu().numpy()[0])
                image_paths.append(str(path))
            except Exception:
                continue

    if not feature_list:
        return np.empty((0, 1024)), []
    return np.asarray(feature_list), image_paths


# =====================================================
# 7. SUPERVISED PREDICTION
# =====================================================
def predict_image(model, image):
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASSES[predicted_index]
    confidence = probabilities[predicted_index].item() * 100
    probability_table = {
        CLASSES[i]: probabilities[i].item() * 100
        for i in range(len(CLASSES))
    }
    return predicted_class, confidence, probability_table, tensor


# =====================================================
# 8. SIMILARITY SEARCH
# =====================================================
def find_similar_images(model, tensor):
    query_feature = extract_features(model, tensor).cpu().numpy()
    bank_features, bank_paths = build_feature_bank()
    if len(bank_paths) == 0:
        return []

    similarities = cosine_similarity(query_feature, bank_features)[0]
    top_indices = np.argsort(similarities)[::-1][:5]
    results = []

    for index in top_indices:
        results.append({
            "path": bank_paths[index],
            "similarity": float(similarities[index]),
        })
    return results


# =====================================================
# 9. HEADER
# =====================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI-assisted retinal imaging • Research workspace</div>
      <h1>OCT-C8 Retinal Image Analysis</h1>
      <p>Explore 8-class retinal OCT image predictions and compare learned image features with reference scans from your training dataset.</p>
      <span class="hero-tag">DenseNet121&nbsp; · &nbsp;8 classes&nbsp; · &nbsp;Feature similarity</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note"><strong>Research use only.</strong> Predictions and similarity scores are model outputs, not a medical diagnosis. The model may misclassify images from other datasets or non-OCT images.</div>',
    unsafe_allow_html=True,
)

# =====================================================
# 10. UPLOAD PANEL
# =====================================================
upload_col, info_col = st.columns([1.45, 1], gap="large")

with upload_col:
    st.markdown('<div class="section-heading">Upload an OCT scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Choose a JPG, PNG, BMP or TIFF image to start analysis.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select image file",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        label_visibility="collapsed",
    )

with info_col:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-label">Workflow overview</div>
          <p style="margin:.25rem 0 .6rem 0;color:#243b53;"><strong>01</strong> &nbsp; Upload an image</p>
          <p style="margin:.25rem 0 .6rem 0;color:#243b53;"><strong>02</strong> &nbsp; Run DenseNet121 classification</p>
          <p style="margin:.25rem 0 .6rem 0;color:#243b53;"><strong>03</strong> &nbsp; Review class probabilities</p>
          <p style="margin:.25rem 0 0 0;color:#243b53;"><strong>04</strong> &nbsp; Compare with reference scans</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================
# 11. IMAGE ANALYSIS
# =====================================================
if uploaded_file is None:
    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:2.5rem 1rem;margin-top:1rem;">
          <div style="font-size:2.4rem;">🖼️</div>
          <h3 style="color:#10233f;margin:.5rem 0;">Ready when you are</h3>
          <p style="color:#64748b;margin:0;">Upload an OCT image above, then run the analysis to view the prediction and feature similarity results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not read this file as an image. Please upload a valid image.")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="section-heading">Image preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Confirm that the selected file is the retinal OCT scan you intend to analyze.</div>', unsafe_allow_html=True)

    preview_col, detail_col = st.columns([1.2, 1], gap="large")
    with preview_col:
        st.image(image, use_container_width=True, caption=uploaded_file.name)
    with detail_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">File details</div>', unsafe_allow_html=True)
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Dimensions:** {image.width} × {image.height} px")
        st.write(f"**Color mode:** RGB")
        st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
        st.markdown("</div>", unsafe_allow_html=True)
        analyze = st.button("Run image analysis", type="primary", use_container_width=True)

    if analyze:
        if not MODEL_PATH.exists():
            st.error(f"Model weights not found: `{MODEL_PATH}`")
            st.stop()

        try:
            with st.spinner("Loading DenseNet121 and analyzing the image..."):
                model = load_model()
                predicted_class, confidence, probability_table, tensor = predict_image(model, image)

            st.markdown("---")
            st.markdown('<div class="section-heading">Classification results</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">The predicted class is the highest-scoring class among the eight labels known to this model.</div>', unsafe_allow_html=True)

            result_col, probability_col = st.columns([0.8, 1.2], gap="large")
            with result_col:
                st.markdown(
                    f"""
                    <div class="panel">
                      <div class="panel-label">Supervised prediction</div>
                      <div class="prediction-class">{html.escape(predicted_class)}</div>
                      <span class="status-pill">DenseNet121 output</span>
                      <div style="height:1rem"></div>
                      <div style="color:#64748b;font-size:.85rem;">Model confidence</div>
                      <div style="font-size:2rem;font-weight:800;color:#10233f;font-family:Manrope,sans-serif;">{confidence:.2f}%</div>
                      <div style="background:#e8eef5;border-radius:999px;height:8px;margin-top:.55rem;overflow:hidden;">
                        <div style="width:{max(0, min(confidence, 100)):.2f}%;height:8px;background:linear-gradient(90deg,#0f9d91,#3ab7a8);border-radius:999px;"></div>
                      </div>
                      <p style="color:#8290a3;font-size:.76rem;margin:.8rem 0 0 0;">Confidence is not a guarantee of correctness, especially for images outside the training distribution.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with probability_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-label">Probability by class</div>', unsafe_allow_html=True)
                probability_df = pd.DataFrame({
                    "Class": list(probability_table.keys()),
                    "Probability (%)": list(probability_table.values()),
                }).sort_values("Probability (%)", ascending=False)
                st.bar_chart(probability_df.set_index("Class"), y="Probability (%)", height=300)
                st.dataframe(
                    probability_df.style.format({"Probability (%)": "{:.2f}%"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-heading">Feature similarity</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">The nearest reference images according to cosine similarity of DenseNet121 features. Similarity is not a disease probability.</div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Comparing features with reference OCT images..."):
                similar_images = find_similar_images(model, tensor)

            if not similar_images:
                st.info("No reference images found. Check that `dataset/train` contains one folder per class with image files.")
            else:
                sim_cols = st.columns(5, gap="medium")
                for rank, result in enumerate(similar_images, start=1):
                    path = Path(result["path"])
                    similarity = result["similarity"] * 100
                    with sim_cols[rank - 1]:
                        st.markdown('<div class="similarity-card">', unsafe_allow_html=True)
                        st.image(str(path), use_container_width=True)
                        st.markdown(f'<div class="rank">Reference {rank}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="sim-score">{similarity:.2f}%</div>', unsafe_allow_html=True)
                        st.markdown(f'<span class="class-tag">{html.escape(path.parent.name)}</span>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="note"><strong>Interpretation reminder:</strong> supervised classification and feature similarity answer different questions, so their class labels can differ. This app does not verify that an uploaded file is a valid OCT scan.</div>',
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")
            with st.expander("Technical error details"):
                st.exception(error)

st.markdown(
    '<div class="footer">OCT-C8 Retinal Image Analysis • Research prototype • DenseNet121</div>',
    unsafe_allow_html=True,
)
