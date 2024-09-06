import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class Qop:
    def __init__(self, dtype, w_a=None, affine='tensor', affine_dim=None, group_size=-1, symentric=False):
        self.symentric = symentric
        self.dtype = dtype

        if self.dtype in [torch.bfloat16, torch.float32, torch.float64, torch.float16]:
            self.q_min = torch.finfo(self.dtype).min
            self.q_max = torch.finfo(self.dtype).max
        else:
            self.q_min = torch.iinfo(self.dtype).min
            self.q_max = torch.iinfo(self.dtype).max

        self.min_val = None
        self.max_val = None

        self.w_a = w_a
        self.q_group_size = group_size
        self.affine = affine
        self.affine_dim = affine_dim

        self.scales = None
        self.zero_point = None

    @torch.no_grad()
    def compute_scales_zero_point(self, tensor=None):
        if self.min_val is not None and self.max_val is not None:
            if self.symentric:
                scales = self.max_val / self.q_max
            else:
                scales = (self.max_val - self.min_val) / (self.q_max - self.q_min)
        else:
            if self.symentric:
                self.max_val = tensor.abs().max().item()
                scales = self.max_val / self.q_max
            else:
                self.max_val = tensor.max().item()
                self.min_val = tensor.min().item()
                scales = (self.max_val - self.min_val) / (self.q_max - self.q_min)

        zero_point = 0 if self.symentric else self.q_min - (self.min_val / scales)
        return scales, zero_point

    @torch.no_grad()
    def compute_scales_zero_point_dimension(self, tensor, dim=-1):
        if dim >= 0:  # Channel-Wise, Group-Wise
            output_dim = tensor.shape[dim]
            scales = torch.zeros(output_dim)
            zero_point = torch.zeros(output_dim)
            for index in range(output_dim):
                sub_tensor = tensor.select(dim, index)
                scales[index], zero_point[index] = self.compute_scales_zero_point(sub_tensor)

            scale_shape = [1] * tensor.dim()
            scale_shape[dim] = -1
            self.scales, self.zero_point = scales.view(scale_shape), zero_point.view(scale_shape)
        else:  # Tensor-Wise
            self.scales, self.zero_point = self.compute_scales_zero_point(tensor)

    def compute_scale_zero_pointer(self):
        if self.affine == 'tensor':
            self.compute_scales_zero_point_dimension(self.tensor)
        elif self.affine == 'channel':
            self.compute_scales_zero_point_dimension(self.tensor, dim=self.affine_dim)
        elif self.affine == 'group':
            assert self.tensor_shape[1] % self.q_group_size == 0, "tensor shape and group size mismatch"
            assert self.tensor.dim() == 2, "tensor dimension mismatch for group quantization"
            tensor = self.tensor.view(-1, self.q_group_size)
            self.compute_scales_zero_point_dimension(tensor, dim=0)

    def push_to_tensor_device(self, tensor):
        if isinstance(self.zero_point, torch.Tensor):
            self.zero_point = self.zero_point.clone().detach().to(tensor.device)
        else:
            self.zero_point = torch.tensor(self.zero_point).to(tensor.device)

    def quantize(self, tensor):
        self.tensor = tensor
        self.tensor_shape = self.tensor.shape

        if self.zero_point is None or self.scales is None:
            self.compute_scale_zero_pointer()

        tensor = self.tensor.detach().clone()
        self.push_to_tensor_device(tensor)

        if self.affine == 'group':
            orig_tensor_shape = tensor.shape
            tensor = tensor.view(tensor.shape[0] * (tensor.shape[1] // self.q_group_size), -1)

        quantized_tensor = torch.round(tensor / self.scales + self.zero_point).clamp(self.q_min, self.q_max) if not self.symentric else torch.round(tensor / self.scales).clamp(self.q_min, self.q_max)

        if self.affine == 'group':
            quantized_tensor = quantized_tensor.view(orig_tensor_shape)

        return quantized_tensor.type(self.dtype)

    @torch.no_grad()
    def dequantize(self, quantized_tensor):
        self.push_to_tensor_device(quantized_tensor)

        if self.affine == 'group':
            quantized_tensor_reshaped = quantized_tensor.view(quantized_tensor.shape[0] * (quantized_tensor.shape[1] // self.q_group_size), -1)
            dequantized_tensor = self.scales * (quantized_tensor_reshaped.float() - self.zero_point)
            self.dequantized_tensor = dequantized_tensor.view(quantized_tensor.shape)
        else:
            self.dequantized_tensor = self.scales * (quantized_tensor.float() - self.zero_point)

        return self.dequantized_tensor

    @torch.no_grad()
    def compute_dequantization_error(self, original_tensor, dequantized_tensor):
        if torch.isinf(original_tensor).any() or torch.isinf(dequantized_tensor).any():
            print("Inf values detected")
        if torch.isnan(original_tensor).any() or torch.isnan(dequantized_tensor).any():
            print("NaN values detected")

        max_value = dequantized_tensor.abs().max()
        if max_value > 0:
            dequantized_tensor /= max_value

        return F.mse_loss(original_tensor, dequantized_tensor)

    def plot_quantization_errors(self, original_tensor, quantized_tensor, dequantized_tensor):
        n_bits = 8
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        self.plot_matrix(original_tensor, axes[0], 'Original Tensor', cmap=ListedColormap(['white']))
        q_min, q_max = torch.iinfo(self.dtype).min, torch.iinfo(self.dtype).max
        self.plot_matrix(quantized_tensor, axes[1], f'{n_bits}-bit Linear Quantized Tensor', vmin=q_min, vmax=q_max, cmap='coolwarm')
        self.plot_matrix(dequantized_tensor, axes[2], 'Dequantized Tensor', cmap='coolwarm')
        q_error_tensor = abs(original_tensor - dequantized_tensor)
        self.plot_matrix(q_error_tensor, axes[3], 'Quantization Error Tensor', cmap=ListedColormap(['white']))

        fig.tight_layout()
        plt.show()

    def plot_matrix(self, tensor, ax, title, vmin=0, vmax=1, cmap=None):
        sns.heatmap(tensor.cpu().numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", cbar=False)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
