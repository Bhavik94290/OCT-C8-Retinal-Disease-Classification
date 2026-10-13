from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
GRADCAM_DIR = ROOT / "gradcam"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

NUM_CLASSES = 8
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

BASELINE_EPOCHS = 5
PROPOSED_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for p in [MODEL_DIR, CHECKPOINT_DIR, RESULTS_DIR, GRADCAM_DIR]:
    p.mkdir(parents=True, exist_ok=True)
