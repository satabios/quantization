import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


class Qop:
    """Quantization operation supporting int8 and bfloat16 data types"""

    def __init__(self, dtype, w_a=None, affine='tensor', affine_dim=None, group_size=-1, symmetric=False):
        self.symmetric = symmetric
        self.dtype = dtype
        info = torch.finfo if dtype.is_floating_point else torch.iinfo
        self.q_min = info(self.dtype).min
        self.q_max = info(self.dtype).max

        self.min_val = None
        self.max_val = None
        self.w_a = w_a
        self.q_group_size = group_size

        # Defines the granularity of quantization: per tensor, channel, or group
        self.affine = affine
        self.affine_dim = affine_dim

        self.scales = None
        self.zero_point = None

    @torch.no_grad()
    def compute_scales_zero_point(self, tensor=None):
        """Computes scale and zero-point based on tensor's min/max values."""
        if self.min_val is not None and self.max_val is not None:
            if self.symmetric:
                scales = self.max_val / self.q_max
            else:
                scales = (self.max_val - self.min_val) / (self.q_max - self.q_min)
        else:
            if self.symmetric:
                self.max_val = tensor.abs().max().item()
                scales = self.max_val / self.q_max
            else:
                self.max_val = tensor.max().item()
                self.min_val = tensor.min().item()
                scales = (self.max_val - self.min_val) / (self.q_max - self.q_min)

        if self.symmetric:
            zero_point = 0
        else:
            zero_point = self.q_min - self.min_val / scales

        return scales, zero_point

    @torch.no_grad()
    def compute_scales_zero_point_dimension(self, tensor, dim=-1):
        """Computes scale and zero-point for different dimensions (e.g., per channel)."""
        if dim >= 0:  # Channel-Wise or Group-Wise
            output_dim = tensor.shape[dim]
            scale = torch.zeros(output_dim)
            zero_point = torch.zeros(output_dim)
            for index in range(output_dim):
                sub_tensor = tensor.select(dim, index)
                scale[index], zero_point[index] = self.compute_scales_zero_point(sub_tensor)

            # Reshape scale and zero_point to match tensor dimensions
            scale_shape = [1] * tensor.dim()
            scale_shape[dim] = output_dim
            self.scales = scale.view(scale_shape)
            self.zero_point = zero_point.view(scale_shape)
        else:  # Tensor-Wise
            self.scales, self.zero_point = self.compute_scales_zero_point(tensor)

    def compute_scale_zero_pointer(self):
        """Computes the scale and zero-point based on the affine type (tensor, channel, or group)."""
        if self.affine == 'tensor':  # Per Tensor
            self.compute_scales_zero_point_dimension(self.tensor)
        elif self.affine == 'channel':  # Per Channel
            self.compute_scales_zero_point_dimension(self.tensor, dim=self.affine_dim)
        elif self.affine == 'group':  # Per Group (only for Linear layers)
            assert self.tensor.shape[1] % self.q_group_size == 0
            assert self.tensor.dim() == 2  # Only for Linear layers

            tensor = self.tensor.view(-1, self.q_group_size)
            self.compute_scales_zero_point_dimension(tensor, dim=0)

    def push_to_tensor_device(self, tensor_device):
        """Moves scale and zero-point to the specified device."""
        if isinstance(self.zero_point, torch.Tensor):
            self.zero_point = self.zero_point.clone().detach().to(tensor_device)
        else:
            self.zero_point = torch.tensor(self.zero_point).to(tensor_device)

        if isinstance(self.scales, torch.Tensor):
            self.scales = self.scales.clone().detach().to(tensor_device)
        else:
            self.scales = torch.tensor(self.scales).to(tensor_device)

    def quantize(self, tensor):
        """Quantizes the input tensor."""
        self.tensor = tensor
        self.tensor_shape = self.tensor.shape

        if self.zero_point is None or self.scales is None:  # If not pre-computed, calculate them
            self.compute_scale_zero_pointer()

        tensor = self.tensor.detach().clone()
        self.push_to_tensor_device(tensor.device)

        if self.affine == 'group':  # Handle group quantization (for Linear layers)
            orig_tensor_shape = tensor.shape
            tensor = tensor.view(tensor.shape[0] * (tensor.shape[1] // self.q_group_size), -1)

        if self.symmetric:
            self.quantized_tensor = torch.round(tensor / self.scales).clamp(self.q_min, self.q_max)
        else:
            self.quantized_tensor = torch.round(tensor / self.scales + self.zero_point).clamp(self.q_min, self.q_max)

        if self.affine == 'group':  # Reshape back after quantization
            self.quantized_tensor = self.quantized_tensor.view(orig_tensor_shape)

        return self.quantized_tensor.type(self.dtype)

    @torch.no_grad()
    def dequantize(self, quantized_tensor, activation=False):
        """Dequantizes the quantized tensor."""
        self.push_to_tensor_device(quantized_tensor.device)

        if self.affine == 'group':  # Handle group dequantization (for Linear layers)
            reshaped_tensor = quantized_tensor.view(
                quantized_tensor.shape[0] * (quantized_tensor.shape[1] // self.q_group_size), -1)
            dequantized_tensor = self.scales * (reshaped_tensor.float() - self.zero_point)
            self.dequantized_tensor = dequantized_tensor.view(quantized_tensor.shape)
        else:
            if activation and self.scales.shape[1] == 1:
                self.zero_point = self.zero_point.view(1, self.zero_point.shape[0], 1, 1)
                self.scales = self.scales.view(1, self.scales.shape[0], 1, 1)

            self.dequantized_tensor = self.scales * (quantized_tensor.float() - self.zero_point)

        return self.dequantized_tensor

    @torch.no_grad()
    def compute_dequantization_error(self, original_tensor, dequantized_tensor):
        """Computes the mean squared error between original and dequantized tensors."""
        if torch.isinf(original_tensor).any() or torch.isinf(dequantized_tensor).any():
            print("Inf values detected")
        if torch.isnan(original_tensor).any() or torch.isnan(dequantized_tensor).any():
            print("NaN values detected")

        # Normalize the tensors to avoid scale issues
        max_value = dequantized_tensor.abs().max()
        if max_value > 0:
            dequantized_tensor /= max_value

        return F.mse_loss(original_tensor, dequantized_tensor)
