import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
print(torch.version.cuda)  # Should show the CUDA version if correctly installed
print(torch.cuda.is_available())  # Should return True
