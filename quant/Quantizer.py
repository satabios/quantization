import torch
from Qop import Qop
import torch.nn as nn
import torch.quantization.observer as observer

class Quantizer(nn.Module):
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
            data_metry = None,
            activations=None

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

        #
        # if(activations is not None): #Only if the activations are given
        #     self.activation_quant = Qop(dtype=data_metry['activations']['dtype'],
        #                                 symentric=data_metry['activations']['symmentry'],
        #                                 affine="tensor",
        #                                 affine_dim=None)
        #
        #
        #
        #
        #     #Custom Observer for Activations
        #     hist_observer = observer.HistogramObserver()
        #     hist_observer(activations)
        #     self.activation_quant.scales, self.activation_quant.zero_point = hist_observer.calculate_qparams()
        #     # deq_a = self.activation_quant.dequantize(activations)
        #     # print(self.activation_quant.compute_dequantization_error(activations, deq_a))

        self.weight_quant = Qop(dtype=data_metry['weights']['dtype'],
                                symentric=data_metry['weights']['symmentry'],
                                affine=data_metry['weights']['affine'],
                                affine_dim=data_metry['weights']['affine_dim'])

        self.quantize_output = quantize_output  # Flag

        if(self.quantize_output):

            self.output_quant = Qop(dtype=data_metry['outputs']['dtype'],
                                    symentric=data_metry['outputs']['symmentry'],
                                    affine="tensor",
                                    affine_dim=None)

            self.output_quant.min_val = data_metry['outputs']['affine']
            self.output_quant.max_val = data_metry['outputs']['affine_dim']
            self.output_quant.scales, self.output_quant.zero_point = self.output_quant.compute_scales_zero_point()

    def to(self, *args, **kwargs):
        super(Quantizer, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):

        self.weight = self.weight if(x.dtype==self.weight.dtype) else self.weight_quant.dequantize(self.weight)

        if (self.weight.dim() == 4):

            y = torch.functional.F.conv2d(input= x,
                                          weight= self.weight,
                                          stride=self.stride,
                                          padding=self.padding,
                                          # bias=self.bias,
                                          dilation=self.dilation,
                                          groups=self.groups)
        else:
            y = torch.functional.F.linear(input= x,
                                          weight=self.weight,
                                          # bias=self.bias
                                          )

        if (self.bias is not None):
            y_index = y.shape.index(self.bias.shape[0])
            reshaped_bias = self.bias.view([1 if i != y_index else self.bias.shape[0] for i in range(len(y.shape))])
            y = y+reshaped_bias

        # Output Quantization
        if  y.dtype != self.weight.dtype:
            return self.output_quant.dequantize(y)
        if self.quantize_output:
            return self.output_quant.quantize(y)
        else:
            return y

    @staticmethod
    def from_float(
            module, weight_quant="tensor", act_quant="tensor", quantize_output=False, activations=None, data_metry=None

    ):


        # Weight per_channel/per_tensor quantization; Activation per_token/per_tensor quantization
        if (isinstance(module, torch.nn.Linear)):
            new_module = Quantizer(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                quantize_output=quantize_output,
                data_metry = data_metry,
                activations=activations,

            )
        elif (isinstance(module, torch.nn.Conv2d)):
            new_module = Quantizer(
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
                data_metry=data_metry,
                activations = activations
            )

        q_weight = new_module.weight_quant.quantize(module.weight)
        # d_weight = new_module.weight_quant.dequantize(q_weight)
        # print(f"{module}: Error \"{new_module.weight_quant.compute_dequantization_error(module.weight, d_weight)}\" ")
        new_module.weight = q_weight

        return new_module

    def __repr__(self):
        output_qdeets = f"outputq={self.data_metry['outputs']['symmentry'][:4]}/{str(self.data_metry['outputs']['dtype']).split('.')[-1]}/{self.data_metry['outputs']['affine']}" if (self.data_metry['outputs']['dtype'] is not None) else f"outputq={None}"
        qdeets = (f" ,weightq={self.data_metry['weights']['symmentry'][:4]}/{str(self.data_metry['weights']['dtype']).split('.')[-1]}/{self.data_metry['weights']['affine']}, {output_qdeets}"
                  )
        if self.cnn:
            return f"QConv2d({self.in_features}, {self.out_features}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}"+qdeets
        else:
            return f"QLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}"+qdeets
