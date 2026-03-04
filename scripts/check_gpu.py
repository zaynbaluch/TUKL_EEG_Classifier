import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Try a simple tensor operation on GPU
    try:
        x = torch.randn(1, 1).cuda()
        print("Successfully created a tensor on CUDA.")
    except Exception as e:
        print(f"CUDA Error during test: {e}")
else:
    print("CUDA is NOT available to PyTorch.")
