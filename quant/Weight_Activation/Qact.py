import torch
from torch import nn
from quant.Qop import Qop

class Qact(nn.Module):
    def __init__(self, qconfig):
        super(Qact, self).__init__()
        self.qconfig = qconfig

        self.act_quant = Qop(
                                dtype=torch.int8,
                                symentric=False,
                                affine='tensor',
                                affine_dim=None )
        self.act_quant.min_val, self.act_quant.max_val = qconfig['min_val'], qconfig['max_val']
        self.act_quant.scales, self.act_quant.zero_point = self.act_quant.compute_scales_zero_point()

    def forward(self, x):
        return self.act_quant.quantize(x)