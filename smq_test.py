from activation_awareness import get_act_scale
import torch

l_w = torch.rand(512,10)
l_c = torch.rand(512,8,32,32)
print(get_act_scale(l_w).shape,get_act_scale(l_c).shape)

# print(torch.allclose(o,l_w.abs().amax(dim=0, keepdim=True)))