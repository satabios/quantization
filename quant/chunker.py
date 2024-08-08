import torch
from quantization import Quantizer
import torch.nn as nn

class Chunker(nn.Module):
    def __init__(
            self,
            in_features,  # C_in/Lin-in (CNN/Linear)
            out_features,  # C_out/Lin-out (CNN/Linear)
            kernel_size=None,
            stride=None,
            padding=None,
            dilation=None,
            groups=None,
            bias=True,
            quantize_output=False,
            cnn=False,
            data_metry = {'weights': {'dtype': torch.int8, 'symmentry': False, 'per':"tensor"},
                          'activations': {'dtype': torch.int8, 'symmentry': False, 'per':"tensor"},
                          'outputs': {'dtype': torch.int8, 'symmentry': False, 'per':"tensor"} }

    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cnn = cnn
        self.data_metry = data_metry
        if self.cnn:
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.weight_shape = (self.out_features, self.in_features, *self.kernel_size)
            # self.data_metry['weights']['per'] = "" Change it to Channel-Wise
        else:
            self.weight_shape = (self.out_features, self.in_features)


        self.register_buffer(
            "weight",
            torch.randn(
                self.weight_shape,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features),
                    dtype=torch.float16,
                    requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.quantize_output = quantize_output #Flag

        self.activation_quant = Quantizer(dtype=data_metry['activations']['dtype'],
                                          symentric=data_metry['activations']['symmentry'],
                                          per=data_metry['activations']['per'])
        self.weight_quant = Quantizer(dtype=  data_metry['weights']['dtype'],
                                     symentric= data_metry['weights']['symmentry'],
                                     per= data_metry['weights']['per'])



        self.output_quant = Quantizer(dtype=data_metry['outputs']['dtype'],
                                          symentric=data_metry['outputs']['symmentry'],
                                          per= data_metry['outputs']['per'])


    def to(self, *args, **kwargs):
        super(Chunker, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):

        # Activation Quantization
        q_x = self.activation_quant.quantize(x)


        # Weights Quantization
        if self.bias is not None:
            self.bias = self.bias.to(self.weight.dtype)
        else:
            self.bias = None

        if (self.weight.dim() == 4):
            y = torch.functional.F.conv2d(q_x,
                                          self.weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups)
        else:
            y = torch.functional.F.linear(q_x,
                                          self.weight,
                                          self.bias)

        # Output Quantization
        if self.quantize_output:
            return self.output_quant.quantize(y)
        else:
            return self.output_quant.dequantize(y)



    @staticmethod
    def from_float(
            module, weight_quant="tensor", act_quant="tensor", quantize_output=False
    ):

        data_metry = {'weights': {'dtype': torch.int8, 'symmentry': False, 'per': weight_quant},
                      'activations': {'dtype': torch.int8, 'symmentry': False, 'per': act_quant},
                      'outputs': {'dtype': torch.int8, 'symmentry': False, 'per': "tensor"}}

        # Weight per_channel/per_tensor quantization; Activation per_token/per_tensor quantization
        if (isinstance(module, torch.nn.Linear)):
            new_module = Chunker(
                module.in_features,
                module.out_features,
                module.bias is not None,
                quantize_output=quantize_output,
                data_metry = data_metry


            )
        elif (isinstance(module, torch.nn.Conv2d)):
            new_module = Chunker(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
                module.bias is not None,
                quantize_output=quantize_output,
                cnn=True,
                data_metry=data_metry
            )


        new_module.weight = new_module.weight_quant.quantize(module.weight)

        if module.bias is not None:
            new_module.bias = new_module.bias_quant.quantize(module.bias)


        return new_module

    def __repr__(self):
        if self.cnn:
            return f"QConv2d({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None} weight_quant={self.data_metry['weights']['per']}, act_quant={self.data_metry['activations']['per']}, output_quant={self.data_metry['outputs']['per']})"
        else:
            return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None} weight_quant={self.data_metry['weights']['per']}, act_quant={self.data_metry['activations']['per']}, output_quant={self.data_metry['outputs']['per']})"
