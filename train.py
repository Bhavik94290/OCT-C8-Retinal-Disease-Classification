import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models import get_model
from src.data import get_loaders


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

CHECKPOINTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_accuracy,
    history,
    model_name
):

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
        "history": history
    }

    temporary = CHECKPOINTS / f"{model_name}_latest.tmp"
    final = CHECKPOINTS / f"{model_name}_latest.pt"

    torch.save(checkpoint, temporary)

    # Atomic replacement
    os.replace(temporary, final)

    print(f"Checkpoint saved: {final}")


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    model_name
):

    path = CHECKPOINTS / f"{model_name}_latest.pt"

    if not path.exists():
        return 0, 0.0, []

    print("\n" + "=" * 60)
    print("CHECKPOINT FOUND")
    print("=" * 60)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_accuracy = checkpoint.get(
        "best_accuracy",
        0.0
    )

    history = checkpoint.get(
        "history",
        []
    )

    print(
        f"Resuming from epoch "
        f"{start_epoch + 1}/{EPOCHS}"
    )

    return start_epoch, best_accuracy, history


# ============================================================
# CSV
# ============================================================

def save_csv(history, model_name):

    if not history:
        return

    path = RESULTS / f"{model_name}_history.csv"

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(
        loader,
        desc="Training",
        leave=False
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = outputs.argmax(1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = outputs.argmax(1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model_name):

    print("\n")
    print("=" * 70)
    print(f"MODEL: {model_name.upper()}")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
    else:
        print("GPU: None - CPU training")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
        classes
    ) = get_loaders(
        DATASET,
        BATCH_SIZE
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = get_model(
        model_name
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------

    (
        start_epoch,
        best_accuracy,
        history
    ) = load_checkpoint(
        model,
        optimizer,
        scheduler,
        model_name
    )

    no_improvement = 0

    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS
    ):

        print("\n" + "=" * 70)
        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )
        print("=" * 70)

        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(val_loss)

        epoch_time = (
            time.time() - start_time
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nTrain Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_acc * 100:.2f}%"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Acc    : {val_acc * 100:.2f}%"
        )

        print(
            f"Learning Rate: {lr:.8f}"
        )

        print(
            f"Epoch Time : {epoch_time / 60:.2f} min"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr,
            "epoch_time_seconds": epoch_time
        })

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        save_csv(
            history,
            model_name
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_acc > best_accuracy:

            best_accuracy = val_acc
            no_improvement = 0

            best_path = (
                CHECKPOINTS /
                f"{model_name}_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "classes": classes
                },
                best_path
            )

            print(
                f"BEST MODEL SAVED: {best_path}"
            )

        else:

            no_improvement += 1

        # ----------------------------------------------------
        # SAVE LATEST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_accuracy,
            history,
            model_name
        )
        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if no_improvement >= PATIENCE:

            print(
                "\nEarly stopping activated."
            )

            break

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    final_path = (
        MODELS /
        f"{model_name}_final.pt"
    )

    torch.save(
        model.state_dict(),
        final_path
    )

    print("\n" + "=" * 70)
    print("MODEL TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Final model: {final_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "vgg19",
            "densenet121",
            "proposed"
        ]
    )

    args = parser.parse_args()

    train_model(
        args.model
    )
