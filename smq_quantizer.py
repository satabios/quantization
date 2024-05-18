import torch
from torch import nn
from functools import partial
import ipdb

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features) : Linear
    # w: (out_channels, in_channels, kernel_size, kernel_size) : Conv2d
    if(w.dim()==4):
        w_copied = copy.deepcopy(w)
        w = w.view(w.shape[0],-1)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)

    if(w.dim()==4):
        scales = scales.view(w_copied.shape[0],1,1,1)
        w = w_copied # Copy back the original weights
    
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8(nn.Module):
    def __init__(
        self,
        in_features,  #C_in
        out_features, #C_out
        kernel_size=None,
        stride=None,
        padding=None,
        dilation=None,
        groups=None,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
        cnn=False,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.cnn = cnn
        self.dtype = torch.float16 if dtype is not None else dtype 
        if cnn:
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.weight_shape = (self.out_features, self.in_features, self.kernel_size, self.kernel_size)
        else:
            self.weight_shape = (self.out_features, self.in_features)
            

        self.register_buffer(
            "weight",
            torch.randn(
                self.weight_shape,
                dtype=self.dtype,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=self.dtype, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        if(self.weight.dim()==4):
            y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        # Weight per_channel/per_tensor quantization; Activation per_token/per_tensor quantization
        if(isinstance(module, torch.nn.Linear)):
            
           
            new_module = W8A8(
                module.in_features,
                module.out_features,
                module.bias is not None,
                act_quant=act_quant,
                quantize_output=quantize_output,
                dtype=module.weight.data.dtype
            )
        elif(isinstance(module, torch.nn.Conv2d)): 
            new_module = W8A8(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                act_quant=act_quant,
                quantize_output=quantize_output,
                dtype = module.weight.data.dtype
            )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        if self.cnn:
            return f"W8A8Conv2d-smq({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
        
        else:
            return f"W8A8Linear-smq({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"
