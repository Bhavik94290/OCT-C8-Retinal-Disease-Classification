import os
import csv
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_loaders
from src.models import ChannelAttention, SpatialAttention


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints" / "ablation"
RESULTS = ROOT / "results" / "ablation"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


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

NUM_CLASSES = 8


# ============================================================
# ABLATION EXPERIMENTS
# ============================================================

EXPERIMENTS = [
    "multiscale",
    "channel",
    "spatial",
    "multiscale_channel",
    "multiscale_spatial",
]


# ============================================================
# MULTI-SCALE BLOCK
# ============================================================

class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels=1024, out_channels=256):

        super().__init__()

        branch_channels = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                branch_channels * 3,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)

        x = torch.cat(
            [b1, b3, b5],
            dim=1
        )

        return self.fusion(x)


# ============================================================
# DENSENET121 ABLATION MODEL
# ============================================================

class AblationDenseNet(nn.Module):

    def __init__(
        self,
        experiment,
        num_classes=8
    ):

        super().__init__()

        from torchvision import models
        from torchvision.models import DenseNet121_Weights

        self.experiment = experiment

        print(
            f"\nCreating DenseNet121 for: "
            f"{experiment.upper()}"
        )

        # ----------------------------------------------------
        # PRETRAINED DENSENET121
        # ----------------------------------------------------

        backbone = models.densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )

        self.features = backbone.features

        # ----------------------------------------------------
        # COMPONENT FLAGS
        # ----------------------------------------------------

        self.use_multiscale = (
            "multiscale" in experiment
        )

        self.use_channel = (
            "channel" in experiment
        )

        self.use_spatial = (
            "spatial" in experiment
        )

        # ----------------------------------------------------
        # FEATURE PROCESSING
        # ----------------------------------------------------

        if self.use_multiscale:

            self.multiscale = MultiScaleBlock(
                in_channels=1024,
                out_channels=256
            )

            feature_channels = 256

        else:

            feature_channels = 1024

        # ----------------------------------------------------
        # CHANNEL ATTENTION
        # ----------------------------------------------------

        if self.use_channel:

            self.channel_attention = ChannelAttention(
                feature_channels
            )

        # ----------------------------------------------------
        # SPATIAL ATTENTION
        # ----------------------------------------------------

        if self.use_spatial:

            self.spatial_attention = SpatialAttention()

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_channels,
                512
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(
                512,
                num_classes
            )
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        x = self.features(x)

        x = torch.relu(x)

        # Multi-scale
        if self.use_multiscale:

            x = self.multiscale(x)

        # Channel attention
        if self.use_channel:

            x = self.channel_attention(x)

        # Spatial attention
        if self.use_spatial:

            x = self.spatial_attention(x)

        # Global average pooling
        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.classifier(x)

        return x


# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def checkpoint_path(experiment):

    return (
        CHECKPOINTS /
        f"{experiment}_latest.pt"
    )


def best_checkpoint_path(experiment):

    return (
        CHECKPOINTS /
        f"{experiment}_best.pt"
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    experiment,
    epoch,
    model,
    optimizer,
    scheduler,
    best_accuracy,
    history
):

    path = checkpoint_path(experiment)

    temp_path = Path(
        str(path) + ".tmp"
    )

    checkpoint = {

        "epoch": epoch,

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "scheduler_state":
            scheduler.state_dict(),

        "best_accuracy":
            best_accuracy,

        "history":
            history,

    }

    torch.save(
        checkpoint,
        temp_path
    )

    os.replace(
        temp_path,
        path
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    experiment,
    model,
    optimizer,
    scheduler
):

    path = checkpoint_path(experiment)

    if not path.exists():

        return 0, 0.0, []

    print("\nCHECKPOINT FOUND")

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

    start_epoch = (
        checkpoint["epoch"] + 1
    )

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
        f"{start_epoch}/{EPOCHS}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    return (
        start_epoch,
        best_accuracy,
        history
    )


# ============================================================
# SAVE HISTORY CSV
# ============================================================

def save_history(
    experiment,
    history
):

    path = (
        RESULTS /
        f"{experiment}_history.csv"
    )

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "learning_rate",
            "epoch_time_minutes"
        ])

        writer.writerows(history)

    return path


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            labels.size(0)
        )

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                labels.size(0)
            )

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss = (
        running_loss / total
    )

    accuracy = (
        correct / total
    )

    return loss, accuracy


# ============================================================
# TEST
# ============================================================

def test_model(
    model,
    loader
):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# ============================================================
# TRAIN ABLATION MODEL
# ============================================================

def run_experiment(
    experiment,
    train_loader,
    val_loader,
    test_loader
):

    print("\n")
    print("=" * 70)
    print(
        f"ABLATION EXPERIMENT: "
        f"{experiment.upper()}"
    )
    print("=" * 70)

    model = AblationDenseNet(
        experiment
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    (
        start_epoch,
        best_accuracy,
        history
    ) = load_checkpoint(
        experiment,
        model,
        optimizer,
        scheduler
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        print("\n")
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        start_time = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer
            )
        )

        val_loss, val_accuracy = (
            validate(
                model,
                val_loader,
                criterion
            )
        )

        scheduler.step(
            val_loss
        )

        epoch_time = (
            time.time() - start_time
        ) / 60

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate   : "
            f"{lr:.7f}"
        )

        print(
            f"Epoch Time      : "
            f"{epoch_time:.2f} minutes"
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history.append([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            lr,
            epoch_time
        ])

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "best_accuracy":
                        best_accuracy,

                    "experiment":
                        experiment,

                    "epoch":
                        epoch
                },
                best_checkpoint_path(
                    experiment
                )
            )

            print(
                f"NEW BEST MODEL!"
            )

            print(
                f"Best Validation Accuracy: "
                f"{best_accuracy * 100:.2f}%"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # SAVE LATEST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            experiment,
            epoch,
            model,
            optimizer,
            scheduler,
            best_accuracy,
            history
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        save_history(
            experiment,
            history
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    best_path = best_checkpoint_path(
        experiment
    )

    checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    # ========================================================
    # TEST
    # ========================================================

    test_accuracy = test_model(
        model,
        test_loader
    )

    print("\n")
    print("=" * 70)

    print(
        f"ABLATION RESULT: "
        f"{experiment.upper()}"
    )

    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result_file = (
        RESULTS /
        f"{experiment}_result.txt"
    )

    with open(
        result_file,
        "w"
    ) as f:

        f.write(
            f"Ablation Experiment: "
            f"{experiment}\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{best_accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%\n"
        )

    return test_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=EXPERIMENTS
    )

    args = parser.parse_args()

    experiment = args.experiment

    print("\n")
    print("=" * 70)
    print("OCT-C8 ABLATION STUDY")
    print("=" * 70)

    print(
        f"Experiment : {experiment}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Epochs     : {EPOCHS}"
    )

    print(
        f"Batch Size : {BATCH_SIZE}"
    )

    print(
        f"Dataset    : {DATASET}"
    )

    print(
        f"Results    : {RESULTS}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
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

    print("\nClasses:")
    print(classes)

    print("\nDataset:")
    print(
        f"Train      : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test       : "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    accuracy = run_experiment(
        experiment,
        train_loader,
        val_loader,
        test_loader
    )

    print("\n")
    print("=" * 70)
    print("ABLATION EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        f"{experiment.upper()} "
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"\nCheckpoint directory:"
    )

    print(
        CHECKPOINTS
    )

    print(
        f"\nResults directory:"
    )

    print(
        RESULTS
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
