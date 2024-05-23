from activation_awareness import get_weight_scale, get_act_scale
import torch
# l_a = torch.rand(10, 512)
# l_c = torch.rand(512, 3, 32,32)
# print(get_act_scale(l_a).shape, get_act_scale(l_c).shape)

l_w = torch.rand(10, 512)
c_w = torch.rand(8, 3, 3,3)
print(get_weight_scale(l_w).shape, get_weight_scale(c_w).shape)





