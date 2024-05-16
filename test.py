import torch

x = torch.randint(-8,8,(3,3))
print(x.shape)
# torch.quantize_per_channel(x,