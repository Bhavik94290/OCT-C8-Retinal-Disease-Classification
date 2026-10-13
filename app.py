import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from src.models import get_model


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="OCT-C8 Image Analysis",
    page_icon="👁️",
    layout="wide"
)

CLASSES = [
    "AMD",
    "CNV",
    "CSR",
    "DME",
    "DR",
    "DRUSEN",
    "MH",
    "NORMAL"
]

DEVICE = torch.device("cpu")

MODEL_PATH = Path("models/densenet121_final.pt")
DATASET_PATH = Path("dataset/train")

# Maximum reference images from each class
IMAGES_PER_CLASS = 20

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

# Keep preprocessing consistent with the existing model
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# 2. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = get_model("densenet121")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# =====================================================
# 3. EXTRACT FEATURES
# =====================================================

def extract_features(model, tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        features = model.model.features(tensor)

        features = F.relu(features)

        features = F.adaptive_avg_pool2d(
            features,
            (1, 1)
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

        features = F.normalize(
            features,
            p=2,
            dim=1
        )

    return features.detach()


# =====================================================
# 4. BUILD REFERENCE FEATURE BANK
# =====================================================

@st.cache_data(show_spinner=False)
def build_feature_bank():

    model = load_model()

    feature_list = []
    image_paths = []

    if not DATASET_PATH.exists():
        return np.empty((0, 1024)), []

    class_directories = sorted(
        p for p in DATASET_PATH.iterdir()
        if p.is_dir()
    )

    progress = st.progress(
        0,
        text="Preparing reference images..."
    )

    total_classes = len(class_directories)

    try:

        for class_index, class_dir in enumerate(
            class_directories
        ):

            # Select at most 20 images from each class
            image_files = sorted(
                p for p in class_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )[:IMAGES_PER_CLASS]

            for path in image_files:

                try:

                    with Image.open(path) as img:
                        reference_image = img.convert("RGB")

                    tensor = TRANSFORM(
                        reference_image
                    ).unsqueeze(0)

                    feature = extract_features(
                        model,
                        tensor
                    )

                    feature_list.append(
                        feature.cpu().numpy()[0]
                    )

                    image_paths.append(str(path))

                except Exception:
                    # Skip an unreadable reference image
                    continue

            progress.progress(
                (class_index + 1) / max(total_classes, 1),
                text=(
                    f"Processing reference class "
                    f"{class_index + 1}/{total_classes}"
                )
            )

    finally:
        progress.empty()

    if not feature_list:
        return np.empty((0, 1024)), []

    return np.asarray(feature_list), image_paths


# =====================================================
# 5. SUPERVISED PREDICTION
# =====================================================

def predict_image(model, image):

    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = CLASSES[predicted_index]

    confidence = (
        probabilities[predicted_index].item() * 100
    )

    probability_table = {
        CLASSES[i]: probabilities[i].item() * 100
        for i in range(len(CLASSES))
    }

    return (
        predicted_class,
        confidence,
        probability_table,
        tensor
    )


# =====================================================
# 6. SIMILARITY SEARCH
# =====================================================

def find_similar_images(model, tensor):

    query_feature = extract_features(
        model,
        tensor
    ).cpu().numpy()

    bank_features, bank_paths = build_feature_bank()

    if len(bank_paths) == 0:

        return []

    similarities = cosine_similarity(
        query_feature,
        bank_features
    )[0]

    # Get the five most similar reference images
    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    results = []

    for index in top_indices:

        results.append({
            "path": bank_paths[index],
            "similarity": float(similarities[index])
        })

    return results


# =====================================================
# 7. USER INTERFACE
# =====================================================

st.title("🩺 OCT-C8 Retinal Image Analysis")

st.write(
    """
    Upload a retinal OCT image to obtain:

    - Supervised classification using trained DenseNet121
    - Feature-based similarity against reference OCT images
    """
)

st.warning(
    "Research demonstration only. "
    "Predictions are not a medical diagnosis."
)


# -----------------------------------------------------
# UPLOAD IMAGE
# -----------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an OCT image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Could not read the uploaded image. "
            "Please upload a valid image."
        )

        st.stop()

    left, right = st.columns(2)

    with left:

        st.subheader("Uploaded OCT Image")

        st.image(
            image,
            width="stretch"
        )


    # -------------------------------------------------
    # ANALYZE BUTTON
    # -------------------------------------------------

    if st.button(
        "Analyze Image",
        type="primary"
    ):

        try:

            if not MODEL_PATH.exists():

                st.error(
                    f"Model weights not found: {MODEL_PATH}"
                )

                st.stop()


            with st.spinner(
                "Loading trained model..."
            ):

                model = load_model()


            # =========================================
            # SUPERVISED PREDICTION
            # =========================================

            with st.spinner(
                "Performing supervised prediction..."
            ):

                (
                    predicted_class,
                    confidence,
                    probability_table,
                    tensor
                ) = predict_image(
                    model,
                    image
                )


            with right:

                st.subheader(
                    "Supervised Prediction"
                )

                st.metric(
                    "Predicted Class",
                    predicted_class
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.write(
                    "Probability for each class"
                )

                st.bar_chart(
                    probability_table
                )


            # =========================================
            # SIMILARITY ANALYSIS
            # =========================================

            st.divider()

            st.subheader(
                "Unsupervised Feature Similarity"
            )

            st.write(
                """
                This section compares learned image features
                against selected reference images from the
                training dataset. Similarity is not a disease
                probability and does not independently diagnose
                a disease.
                """
            )


            with st.spinner(
                "Comparing OCT image features..."
            ):

                similar_images = find_similar_images(
                    model,
                    tensor
                )


            if not similar_images:

                st.info(
                    "No reference images found. "
                    "Please check dataset/train and its class folders."
                )

            else:

                st.write(
                    "Top 5 most similar reference images"
                )

                columns = st.columns(5)

                for rank, result in enumerate(
                    similar_images
                ):

                    path = Path(
                        result["path"]
                    )

                    similarity = (
                        result["similarity"] * 100
                    )

                    with columns[rank]:

                        st.image(
                            str(path),
                            use_container_width=True
                        )

                        st.write(
                            f"**Rank {rank + 1}**"
                        )

                        st.write(
                            f"Similarity: {similarity:.2f}%"
                        )

                        st.caption(
                            f"Reference class: "
                            f"{path.parent.name}"
                        )


            st.success(
                "Image analysis completed."
            )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

            with st.expander(
                "Technical error details"
            ):

                st.exception(error)


else:

    st.info(
        "Upload an OCT image to begin analysis."
    )


# =====================================================
# END OF APPLICATION
# =====================================================
