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
            bias=None,
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

        if bias is not None:
            self.bias = bias
        else:
            self.register_buffer("bias", None)

        self.quantize_output = quantize_output #Flag

        self.activation_quant = Quantizer(dtype=data_metry['activations']['dtype'],
                                          symentric=data_metry['activations']['symmentry'],
                                          per=data_metry['activations']['per'])
        self.weight_quant = Quantizer(dtype=  data_metry['weights']['dtype'],
                                     symentric= data_metry['weights']['symmentry'],
                                     per= data_metry['weights']['per'])

        if(not quantize_output):

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

        if (self.weight.dim() == 4):
            y = torch.functional.F.conv2d(input=q_x,
                                          weight=self.weight,
                                          stride=self.stride,
                                          padding=self.padding,
                                          dilation=self.dilation,
                                          groups=self.groups)
        else:
            y = torch.functional.F.linear(input=q_x,
                                          weight=self.weight
                                          )

        # HANDLE THE BIAS!!! This would bump the output from the quantized dtype to the original dtype

        if (self.bias is not None) or (self.bias is None and self.quantize_output is False):
            if self.bias is None: # If not bias is found yet you want the output to be not quantized
                self.bias = torch.zeros(y.shape[1])
            y_index = y.shape.index(self.bias.shape[0])
            reshaped_bias = self.bias.view([1 if i != y_index else self.bias.shape[0] for i in range(len(y.shape))])
            y = y+reshaped_bias

        # Output Quantization
        if self.quantize_output:
            return self.output_quant.quantize(y)
        else:
            return y

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
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                quantize_output=quantize_output,
                data_metry = data_metry


            )
        elif (isinstance(module, torch.nn.Conv2d)):
            new_module = Chunker(
                in_features=module.in_channels,
                out_features=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias,
                quantize_output=quantize_output,
                cnn=True,
                data_metry=data_metry
            )

        new_module.weight = new_module.weight_quant.quantize(module.weight)

        return new_module

    def __repr__(self):
        if self.cnn:
            return f"QConv2d({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None} weight_quant={self.data_metry['weights']['per']}, act_quant={self.data_metry['activations']['per']}, output_quant={self.data_metry['outputs']['per']})"
        else:
            return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None} weight_quant={self.data_metry['weights']['per']}, act_quant={self.data_metry['activations']['per']}, output_quant={self.data_metry['outputs']['per']})"
