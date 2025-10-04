import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('is_cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
