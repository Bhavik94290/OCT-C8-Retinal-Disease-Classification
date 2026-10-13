from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 224


def get_loaders(dataset_root, batch_size=8):

    dataset_root = Path(dataset_root)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    for directory in [train_dir, val_dir, test_dir]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {directory}"
            )

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )

    print("\nClasses:")
    print(train_dataset.classes)

    print("\nDataset:")
    print(f"Train      : {len(train_dataset)}")
    print(f"Validation : {len(val_dataset)}")
    print(f"Test       : {len(test_dataset)}")

    kwargs = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": False
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **kwargs
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )
