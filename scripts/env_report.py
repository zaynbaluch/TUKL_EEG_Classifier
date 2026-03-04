import torch
import sys
import os

print("---BEGIN ENV REPORT---")
print(f"PYTHON_VER: {sys.version.split()[0]}")
print(f"TORCH_VER: {torch.__version__}")
print(f"CUDA_AVAIL: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"TORCH_CUDA_VER: {torch.version.cuda}")
    print(f"GPU_NAME: {torch.cuda.get_device_name(0)}")
    print(f"GPU_MEM_TOTAL: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Check some common packages
try:
    import subprocess
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    pkgs = [l for l in result.stdout.split('\n') if any(x in l.lower() for x in ['torch', 'cuda', 'cudnn'])]
    print("RELEVANT_PKGS:")
    for p in pkgs:
        print(f"  {p}")
except:
    pass
print("---END ENV REPORT---")
