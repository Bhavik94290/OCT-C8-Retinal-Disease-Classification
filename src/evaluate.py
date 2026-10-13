import torch
import numpy as np
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

from src.models import get_model
from src.data import get_loaders


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "resnet50",
    "vgg19",
    "densenet121",
    "proposed"
]


# ============================================================
# PRINT INFORMATION
# ============================================================

print("=" * 70)
print("OCT-C8 MULTI-MODEL TEST SET EVALUATION")
print("=" * 70)

print(f"Device : {DEVICE}")
print(f"Dataset: {DATASET}")
print(f"Results: {RESULTS}")


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train_loader, val_loader, test_loader, classes = get_loaders(
    DATASET,
    BATCH_SIZE
)


print("\nClasses:")
print(classes)

print("\nDataset:")
print(f"Train      : {len(train_loader.dataset)}")
print(f"Validation : {len(val_loader.dataset)}")
print(f"Test       : {len(test_loader.dataset)}")


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name):

    print("\n")
    print("=" * 70)
    print(f"EVALUATING: {model_name.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECKPOINT PATH
    # --------------------------------------------------------

    checkpoint_path = CHECKPOINTS / f"{model_name}_best.pt"

    if not checkpoint_path.exists():

        print(
            f"WARNING: Checkpoint not found:"
        )
        print(checkpoint_path)

        return None

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("Creating model...")

    model = get_model(model_name)

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    # --------------------------------------------------------
    # BEST VALIDATION ACCURACY
    # --------------------------------------------------------

    best_val_accuracy = checkpoint.get(
        "best_accuracy",
        0.0
    )

    print(
        f"Loaded checkpoint: {checkpoint_path}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TEST SET PREDICTION
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    correct = 0
    total = 0

    print("\nRunning test set evaluation...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Predicted class
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Count correct predictions
            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # Store labels
            all_labels.extend(
                labels.cpu().numpy()
            )

            # Store predictions
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # --------------------------------------------------------
    # TEST ACCURACY
    # --------------------------------------------------------

    test_accuracy = correct / total

    print("\nTEST RESULTS")
    print("-" * 40)

    print(
        f"Correct Predictions : {correct}"
    )

    print(
        f"Total Test Images   : {total}"
    )

    print(
        f"Test Accuracy       : "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # --------------------------------------------------------

    report_path = (
        RESULTS /
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w") as f:

        f.write(
            f"Model: {model_name.upper()}\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX AS IMAGE
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name.upper()} Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        len(classes)
    )

    plt.xticks(
        tick_marks,
        classes,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        classes
    )

    # Put numbers inside matrix
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white"
                if cm[i, j] > threshold
                else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()

    matrix_path = (
        RESULTS /
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved:"
    )

    print(matrix_path)

    print(
        f"Classification report saved:"
    )

    print(report_path)

    return test_accuracy


# ============================================================
# EVALUATE ALL FOUR MODELS
# ============================================================

results = {}

for model_name in MODELS:

    accuracy = evaluate_model(
        model_name
    )

    if accuracy is not None:

        results[model_name] = accuracy


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST ACCURACY COMPARISON")
print("=" * 70)

for model_name, accuracy in results.items():

    print(
        f"{model_name.upper():15s} : "
        f"{accuracy * 100:.2f}%"
    )

print("=" * 70)


# ============================================================
# BEST MODEL
# ============================================================

if len(results) > 0:

    best_model = max(
        results,
        key=results.get
    )

    best_accuracy = results[best_model]

    print(
        f"\nBEST MODEL: {best_model.upper()}"
    )

    print(
        f"BEST TEST ACCURACY: "
        f"{best_accuracy * 100:.2f}%"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)

if len(results) == 4:

    print(
        "ALL FOUR MODELS EVALUATED SUCCESSFULLY"
    )

else:

    print(
        f"{len(results)} MODELS EVALUATED"
    )

print("=" * 70)
