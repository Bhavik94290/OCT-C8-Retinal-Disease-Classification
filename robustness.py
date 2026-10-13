import torch
import numpy as np
import pandas as pd

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFilter

from src.models import get_model


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MODELS = [
    "resnet50",
    "vgg19",
    "densenet121",
    "proposed"
]


CLASS_NAMES = [
    "AMD",
    "CNV",
    "CSR",
    "DME",
    "DR",
    "DRUSEN",
    "MH",
    "NORMAL"
]


# ============================================================
# NORMALIZATION
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


# ============================================================
# BASE TRANSFORM
# ============================================================

base_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


# ============================================================
# ROBUSTNESS CONDITIONS
# ============================================================

CONDITIONS = [
    "Original",
    "Gaussian Noise",
    "Gaussian Blur",
    "Brightness +20%",
    "Brightness -20%",
    "Contrast +20%",
    "Contrast -20%"
]


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dir = DATASET / "test"

if not test_dir.exists():

    raise FileNotFoundError(
        f"Test directory not found: {test_dir}"
    )


test_dataset = datasets.ImageFolder(
    test_dir,
    transform=base_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=False
)


print("=" * 70)
print("OCT-C8 ROBUSTNESS EVALUATION")
print("=" * 70)

print(
    f"Device : {DEVICE}"
)

print(
    f"Dataset: {test_dir}"
)

print(
    f"Test images: {len(test_dataset)}"
)

print(
    f"Classes: {test_dataset.classes}"
)

print("=" * 70)


# ============================================================
# IMAGE CORRUPTIONS
# ============================================================

def apply_corruption(
    images,
    condition
):
    """
    Input:
        images: Tensor [B, 3, H, W]
                values between 0 and 1

    Output:
        corrupted images in [0, 1]
    """

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    if condition == "Original":

        return images.clone()


    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    elif condition == "Gaussian Noise":

        noise = torch.randn_like(
            images
        ) * 0.05

        corrupted = images + noise

        return torch.clamp(
            corrupted,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # GAUSSIAN BLUR
    # --------------------------------------------------------

    elif condition == "Gaussian Blur":

        blur = transforms.GaussianBlur(
            kernel_size=7,
            sigma=2.0
        )

        return blur(images)


    # --------------------------------------------------------
    # BRIGHTNESS +20%
    # --------------------------------------------------------

    elif condition == "Brightness +20%":

        corrupted = images * 1.20

        return torch.clamp(
            corrupted,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # BRIGHTNESS -20%
    # --------------------------------------------------------

    elif condition == "Brightness -20%":

        corrupted = images * 0.80

        return torch.clamp(
            corrupted,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # CONTRAST +20%
    # --------------------------------------------------------

    elif condition == "Contrast +20%":

        mean = images.mean(
            dim=(2, 3),
            keepdim=True
        )

        corrupted = (
            (images - mean) * 1.20
            + mean
        )

        return torch.clamp(
            corrupted,
            0.0,
            1.0
        )


    # --------------------------------------------------------
    # CONTRAST -20%
    # --------------------------------------------------------

    elif condition == "Contrast -20%":

        mean = images.mean(
            dim=(2, 3),
            keepdim=True
        )

        corrupted = (
            (images - mean) * 0.80
            + mean
        )

        return torch.clamp(
            corrupted,
            0.0,
            1.0
        )


    else:

        raise ValueError(
            f"Unknown condition: {condition}"
        )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_images(images):

    mean = MEAN.to(
        images.device
    )

    std = STD.to(
        images.device
    )

    return (
        images - mean
    ) / std


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name):

    print(
        f"\nLoading {model_name.upper()}..."
    )

    model = get_model(
        model_name
    )

    checkpoint_path = (
        CHECKPOINTS /
        f"{model_name}_best.pt"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"Checkpoint loaded:"
    )

    print(
        checkpoint_path
    )

    return model


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name
):

    model = load_model(
        model_name
    )

    model_results = {}

    for condition in CONDITIONS:

        print("\n" + "-" * 60)

        print(
            f"Model     : "
            f"{model_name.upper()}"
        )

        print(
            f"Condition : "
            f"{condition}"
        )

        print("-" * 60)

        correct = 0
        total = 0

        # ----------------------------------------------------
        # EVALUATE TEST SET
        # ----------------------------------------------------

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )

                # Apply corruption
                corrupted_images = (
                    apply_corruption(
                        images,
                        condition
                    )
                )

                # Normalize after corruption
                inputs = normalize_images(
                    corrupted_images
                )

                # Prediction
                outputs = model(
                    inputs
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = (
            correct / total
        ) * 100.0

        model_results[
            condition
        ] = accuracy

        print(
            f"Accuracy: "
            f"{accuracy:.2f}%"
        )

    return model_results


# ============================================================
# RUN ALL MODELS
# ============================================================

all_results = []


for model_name in MODELS:

    try:

        results = evaluate_model(
            model_name
        )

        for condition, accuracy in results.items():

            all_results.append({
                "Model": model_name.upper(),
                "Condition": condition,
                "Accuracy (%)": round(
                    accuracy,
                    2
                )
            })

    except Exception as e:

        print(
            f"\nERROR while evaluating "
            f"{model_name.upper()}"
        )

        print(e)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE LONG-FORM RESULTS
# ============================================================

csv_path = (
    RESULTS /
    "robustness_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

if not df.empty:

    comparison = df.pivot(
        index="Model",
        columns="Condition",
        values="Accuracy (%)"
    )

    # Reorder columns
    comparison = comparison.reindex(
        columns=CONDITIONS
    )

    comparison_path = (
        RESULTS /
        "robustness_comparison.csv"
    )

    comparison.to_csv(
        comparison_path
    )

    # --------------------------------------------------------
    # PRINT TABLE
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)

    print(
        "ROBUSTNESS ACCURACY COMPARISON"
    )

    print("=" * 100)

    print(
        comparison.to_string()
    )

    print("=" * 100)

    # --------------------------------------------------------
    # FIND BEST MODEL FOR EACH CONDITION
    # --------------------------------------------------------

    print("\nBEST MODEL FOR EACH CONDITION")
    print("=" * 60)

    for condition in CONDITIONS:

        if condition not in comparison.columns:
            continue

        column = comparison[
            condition
        ].dropna()

        if len(column) == 0:
            continue

        best_model = column.idxmax()
        best_accuracy = column.max()

        print(
            f"{condition:20s} : "
            f"{best_model:12s} "
            f"({best_accuracy:.2f}%)"
        )

    # --------------------------------------------------------
    # ROBUSTNESS DROP FROM ORIGINAL
    # --------------------------------------------------------

    print("\nROBUSTNESS DROP FROM ORIGINAL")
    print("=" * 70)

    for model in comparison.index:

        original_accuracy = comparison.loc[
            model,
            "Original"
        ]

        print(
            f"\n{model}"
        )

        for condition in CONDITIONS:

            if condition == "Original":
                continue

            accuracy = comparison.loc[
                model,
                condition
            ]

            drop = (
                original_accuracy
                - accuracy
            )

            print(
                f"  {condition:20s}: "
                f"{drop:.2f}% drop"
            )

    # --------------------------------------------------------
    # MOST ROBUST MODEL
    # --------------------------------------------------------

    robustness_conditions = [
        c for c in CONDITIONS
        if c != "Original"
    ]

    mean_robustness = comparison[
        robustness_conditions
    ].mean(
        axis=1
    )

    best_robust_model = (
        mean_robustness.idxmax()
    )

    best_robust_score = (
        mean_robustness.max()
    )

    print("\n")
    print("=" * 70)

    print(
        "OVERALL ROBUSTNESS RESULT"
    )

    print("=" * 70)

    print(
        f"Most robust model: "
        f"{best_robust_model}"
    )

    print(
        f"Average corrupted-image accuracy: "
        f"{best_robust_score:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path = (
        RESULTS /
        "robustness_summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            "OCT-C8 ROBUSTNESS EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            comparison.to_string()
        )

        f.write(
            "\n\n"
        )

        f.write(
            "BEST MODEL FOR EACH CONDITION\n"
        )

        f.write(
            "=" * 60
            + "\n"
        )

        for condition in CONDITIONS:

            if condition not in comparison.columns:
                continue

            column = comparison[
                condition
            ].dropna()

            if len(column) == 0:
                continue

            best_model = column.idxmax()
            best_accuracy = column.max()

            f.write(
                f"{condition}: "
                f"{best_model} "
                f"({best_accuracy:.2f}%)\n"
            )

        f.write(
            "\nOVERALL ROBUSTNESS\n"
        )

        f.write(
            "=" * 60
            + "\n"
        )

        f.write(
            f"Most robust model: "
            f"{best_robust_model}\n"
        )

        f.write(
            f"Average corrupted-image accuracy: "
            f"{best_robust_score:.2f}%\n"
        )


    print(
        f"\nCSV saved:"
    )

    print(
        csv_path
    )

    print(
        f"\nComparison saved:"
    )

    print(
        comparison_path
    )

    print(
        f"\nSummary saved:"
    )

    print(
        summary_path
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)

print(
    "ROBUSTNESS EVALUATION COMPLETED"
)

print("=" * 70)
