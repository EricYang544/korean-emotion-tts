"""Run on the intended training host before installing alignment assets."""
import platform
import subprocess
import sys
import torch
print('Python:',platform.python_version(),sys.executable)
print('Torch:',torch.__version__,'CUDA:',torch.version.cuda)
if not torch.cuda.is_available():raise SystemExit('CUDA GPU is required for training on Vast.ai.')
x=torch.randn(512,512,device='cuda');y=x@x.T;torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('GPU:',torch.cuda.get_device_name(0),'BF16:',torch.cuda.is_bf16_supported())
subprocess.run([sys.executable,'-m','pip','check'],check=True)
