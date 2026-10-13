import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from src.models import get_model


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CHECKPOINTS = ROOT / "checkpoints"
GRADCAM_DIR = ROOT / "gradcam"

GRADCAM_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
NUM_CLASSES = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


MODELS = [
    "resnet50",
    "vgg19",
    "densenet121",
    "proposed"
]


# OCT-C8 classes
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
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    last_conv = None

    for layer in model.modules():

        if isinstance(layer, nn.Conv2d):
            last_conv = layer

    if last_conv is None:
        raise RuntimeError(
            "No Conv2d layer found in model."
        )

    return last_conv


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    # --------------------------------------------------------
    # SAVE ACTIVATION
    # --------------------------------------------------------

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    # --------------------------------------------------------
    # SAVE GRADIENT
    # --------------------------------------------------------

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # GENERATE CAM
    # --------------------------------------------------------

    def generate(self, image_tensor, target_class):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[:, target_class]

        score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        # ----------------------------------------------------
        # GET ACTIVATIONS AND GRADIENTS
        # ----------------------------------------------------

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING OF GRADIENTS
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ----------------------------------------------------
        # WEIGHTED ACTIVATION MAP
        # ----------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        cam = torch.relu(cam)

        # ----------------------------------------------------
        # RESIZE TO 224 x 224
        # ----------------------------------------------------

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # ----------------------------------------------------
        # NORMALIZE 0-1
        # ----------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    # --------------------------------------------------------
    # REMOVE HOOKS
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_name):

    print(
        f"\nLoading model: "
        f"{model_name.upper()}"
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

    model = model.to(DEVICE)

    model.eval()

    print(
        f"Checkpoint loaded: "
        f"{checkpoint_path}"
    )

    return model


# ============================================================
# DENORMALIZE IMAGE
# ============================================================

def denormalize(tensor):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = tensor.cpu() * std + mean

    image = torch.clamp(
        image,
        0,
        1
    )

    image = image.permute(
        1,
        2,
        0
    ).numpy()

    return image


# ============================================================
# CREATE HEATMAP OVERLAY
# ============================================================

def create_overlay(
    original_image,
    cam
):

    heatmap = plt.get_cmap(
        "jet"
    )(cam)[:, :, :3]

    overlay = (
        0.5 * original_image +
        0.5 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return heatmap, overlay


# ============================================================
# FIND ONE TEST IMAGE PER CLASS
# ============================================================

def get_one_image_per_class():

    test_dir = DATASET / "test"

    if not test_dir.exists():

        raise FileNotFoundError(
            f"Test directory not found: "
            f"{test_dir}"
        )

    dataset = datasets.ImageFolder(
        test_dir
    )

    print("\nDetected classes:")
    print(dataset.classes)

    selected = {}

    for path, class_index in dataset.samples:

        if class_index not in selected:

            selected[class_index] = path

    return selected


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam_for_model(
    model_name,
    selected_images
):

    model = load_model(
        model_name
    )

    # --------------------------------------------------------
    # FIND TARGET LAYER
    # --------------------------------------------------------

    target_layer = find_last_conv_layer(
        model
    )

    print(
        "Grad-CAM target layer:"
    )

    print(
        target_layer
    )

    cam_generator = GradCAM(
        model,
        target_layer
    )

    model_output_dir = (
        GRADCAM_DIR /
        model_name
    )

    model_output_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # PROCESS EACH CLASS
    # --------------------------------------------------------

    for class_index in range(
        NUM_CLASSES
    ):

        if class_index not in selected_images:

            print(
                f"Image not found for "
                f"class {class_index}"
            )

            continue

        image_path = (
            selected_images[
                class_index
            ]
        )

        true_class = CLASS_NAMES[
            class_index
        ]

        print(
            f"\nProcessing:"
            f" {true_class}"
        )

        print(
            f"Image:"
            f" {image_path}"
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        original = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            original
        )

        original_display = denormalize(
            image_tensor
        )

        image_tensor = image_tensor.unsqueeze(
            0
        ).to(DEVICE)

        # ----------------------------------------------------
        # GET PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = output.argmax(
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()

        predicted_name = CLASS_NAMES[
            predicted_class
        ]

        print(
            f"True class      : "
            f"{true_class}"
        )

        print(
            f"Predicted class : "
            f"{predicted_name}"
        )

        print(
            f"Confidence      : "
            f"{confidence * 100:.2f}%"
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        cam = cam_generator.generate(
            image_tensor,
            predicted_class
        )

        # ----------------------------------------------------
        # CREATE HEATMAP
        # ----------------------------------------------------

        heatmap, overlay = create_overlay(
            original_display,
            cam
        )

        # ----------------------------------------------------
        # SAVE FIGURE
        # ----------------------------------------------------

        plt.figure(
            figsize=(15, 5)
        )

        # Original image
        plt.subplot(
            1,
            3,
            1
        )

        plt.imshow(
            original_display
        )

        plt.title(
            f"Original\n"
            f"True: {true_class}"
        )

        plt.axis("off")

        # Heatmap
        plt.subplot(
            1,
            3,
            2
        )

        plt.imshow(
            heatmap
        )

        plt.title(
            "Grad-CAM Heatmap"
        )

        plt.axis("off")

        # Overlay
        plt.subplot(
            1,
            3,
            3
        )

        plt.imshow(
            overlay
        )

        plt.title(
            f"Prediction: {predicted_name}\n"
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        plt.axis("off")

        plt.tight_layout()

        output_path = (
            model_output_dir /
            f"{class_index:02d}_{true_class}_gradcam.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )

    cam_generator.remove_hooks()

    print(
        f"\nGrad-CAM completed for "
        f"{model_name.upper()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCT-C8 GRAD-CAM EXPLAINABILITY")
    print("=" * 70)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Dataset: {DATASET}"
    )

    print(
        f"Output : {GRADCAM_DIR}"
    )

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{DATASET}"
        )

    # --------------------------------------------------------
    # GET ONE IMAGE PER CLASS
    # --------------------------------------------------------

    selected_images = (
        get_one_image_per_class()
    )

    print(
        f"\nSelected "
        f"{len(selected_images)} "
        f"test images."
    )

    # --------------------------------------------------------
    # RUN ALL MODELS
    # --------------------------------------------------------

    for model_name in MODELS:

        try:

            generate_gradcam_for_model(
                model_name,
                selected_images
            )

        except Exception as e:

            print(
                f"\nERROR in "
                f"{model_name.upper()}:"
            )

            print(e)

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "GRAD-CAM PROCESS COMPLETED"
    )

    print("=" * 70)

    print(
        f"Results saved in:"
    )

    print(
        GRADCAM_DIR
    )


if __name__ == "__main__":

    main()
