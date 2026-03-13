"""
Run this on the GPU laptop to verify everything is installed correctly.
Usage: python verify_setup.py
"""

import sys


def check_python():
    version = sys.version
    print(f"[OK] Python: {version}")


def check_pytorch():
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")
    except ImportError:
        print("[FAIL] PyTorch not installed. Run: pip install torch")
        return False
    return True


def check_cuda():
    import torch

    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available to PyTorch")
        print(f"       PyTorch CUDA version: {torch.version.cuda}")
        print("       Check that NVIDIA drivers and CUDA toolkit are installed")
        print("       and that PyTorch was installed with the matching CUDA version")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[OK] CUDA available: {torch.version.cuda}")
    print(f"[OK] GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    return True


def check_gpu_compute():
    import torch

    # Create a tensor on GPU and do a simple operation
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y  # matrix multiplication
    print(f"[OK] GPU compute works (matrix multiply result shape: {z.shape})")
    return True


def check_torchvision():
    try:
        import torchvision
        print(f"[OK] torchvision: {torchvision.__version__}")
    except ImportError:
        print("[FAIL] torchvision not installed. Run: pip install torchvision")
        return False
    return True


def check_other_deps():
    deps = ["matplotlib", "numpy", "tqdm"]
    all_ok = True
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            print(f"[OK] {dep}: {version}")
        except ImportError:
            print(f"[FAIL] {dep} not installed")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    print("=" * 50)
    print("ML Learning Environment Verification")
    print("=" * 50)
    print()

    check_python()
    print()

    if not check_pytorch():
        sys.exit(1)

    if not check_cuda():
        print("\nYou can still learn PyTorch basics on CPU, but training will be slow.")
    else:
        print()
        check_gpu_compute()

    print()
    check_torchvision()
    print()
    check_other_deps()

    print()
    print("=" * 50)
    print("Setup verification complete!")
    print("=" * 50)
