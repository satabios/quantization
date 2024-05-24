from activation_awareness import  get_act_scale , smq_scale
import torch
l_a = torch.rand(10, 512)
l_c = torch.rand(512, 3, 32,32)
print(get_act_scale(l_a).shape, get_act_scale(l_c).shape)

# l_w = torch.rand(10, 512)
# c_w = torch.rand(8, 3, 3,3)
# print(smq_scale(l_w).shape, smq_scale(c_w).shape)





