import torch
print('torch', torch.__version__)
try:
    import xformers
    print('xformers', xformers.__version__)
except Exception as e:
    print('xformers import failed:', e)
print('cuda', torch.version.cuda)
