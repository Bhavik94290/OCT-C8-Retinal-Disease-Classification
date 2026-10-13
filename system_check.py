import torch

print("=" * 60)
print("OCT-C8 SYSTEM CHECK")
print("=" * 60)

print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version    :", torch.version.cuda)
    print("GPU             :", torch.cuda.get_device_name(0))
    print("GPU count       :", torch.cuda.device_count())

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU VRAM        : {gpu_memory:.2f} GB")
else:
    print("GPU             : Not available")
    print("Training will use CPU.")

print("=" * 60)
