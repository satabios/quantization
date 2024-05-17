import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class Quantizer(nn.Module):
    def __init__(self, tensor, dtype, w_a=None, per='tensor',per_dim=None, group_size=-1, symentric=False):
        super().__init__()
        self.tensor = tensor
        self.symentric = symentric
        self.dtype = dtype
        # Symmetric or Asymmetric
        self.q_min = torch.iinfo(self.dtype).min
        self.q_max = torch.iinfo(self.dtype).max

        self.w_a = w_a
        self.q_group_size = group_size
        self.tensor_shape = self.tensor.shape
        # Per - Channel or Row or Column or Group
        self.per = per  # Wise --> Tensor, Channel, Group
        self.per_dim = per_dim

        self.compute_scale_zero_pointer()

    def compute_scales(self,tensor):
        self.max_val = tensor.abs().max().item()
        if(self.symentric):
            scales = self.max_val/self.q_max
        else:
            self.min_val = tensor.min().item()
            scales = (self.max_val - self.min_val) / (self.q_max - self.q_min)
        return scales

    def compute_scales_dimension(self, tensor, dim=-1):
        if(dim>=0): # Row, Col, Group
            output_dim = tensor.shape[dim]
            scale = torch.zeros(output_dim)
            for index in range(output_dim):
                sub_tensor = tensor.select(dim, index)
                scale[index] = self.compute_scales(sub_tensor)
            # reshape the scale
            scale_shape = [1] * tensor.dim()
            scale_shape[dim] = -1
            self.scales = scale.view(scale_shape)
        else:
            self.scales =  self.compute_scales(tensor)

    def compute_scale_zero_pointer(self):

        if len(self.tensor_shape) == 2:  # Linear Layer
            if (self.per == 'tensor'):   # Per Tensor
                self.compute_scales_dimension(self.tensor)
            elif(self.per == 'dim'):     # Per Row or Col
                self.compute_scales_dimension(self.tensor, dim=self.per_dim)
            elif(self.per == 'group'):   # Per Group
                assert self.tensor_shape[1] % self.q_group_size == 0
                assert self.tensor.dim() == 2 #For Linear
                tensor = self.tensor.view(-1, self.q_group_size)
                self.compute_scales_dimension(tensor, dim=0)


        if self.symentric:
            self.zero_point = 0

        else:
            self.zero_point = self.q_min - (self.min_val / self.scales)
     
            if(self.per_dim is not None):
                # clip the zero_point to fall in [quantized_min, quantized_max]
                zero_point_broadcasted = torch.ones_like(self.zero_point) * self.zero_point

                self.zero_point = torch.where(
                                                self.zero_point < self.q_min,
                                                self.q_min,
                                                torch.where(
                                                    self.zero_point > self.q_max,
                                                    self.q_max,
                                                    torch.round(zero_point_broadcasted).to(torch.int8)
                                                )
                                            )
            else:
     
                if self.zero_point < self.q_min:
                    self.zero_point = self.q_min
                elif self.zero_point > self.q_max:
                    self.zero_point = self.q_max
                else:
                    # round and cast to int
                    self.zero_point = int(round(self.zero_point))


    def quantize(self):
        tensor = self.tensor.clone()
        self.q_min = torch.iinfo(self.dtype).min
        self.quantized_tensor = torch.round(tensor / self.scales + self.zero_point).clamp(self.q_min, self.q_max)
        return self.quantized_tensor.type(self.dtype)
    def dequantize(self, quantized_tensor):
        self.dequantized_tensor = self.scales * (quantized_tensor.float() - self.zero_point)
        return self.dequantized_tensor




    def plot_matrix(self, tensor, ax, title, vmin=0, vmax=1, cmap=None):
        sns.heatmap(tensor.cpu().numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", cbar=False)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    def plot_quantization_errors(self, original_tensor, quantized_tensor, dequantized_tensor):
        n_bits = 8
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        self.plot_matrix(original_tensor, axes[0], 'Original Tensor', cmap=ListedColormap(['white']))
        q_min, q_max = torch.iinfo(self.dtype).min, torch.iinfo(self.dtype).max
        self.plot_matrix(quantized_tensor, axes[1], f'{n_bits}-bit Linear Quantized Tensor', vmin=q_min, vmax=q_max,
                    cmap='coolwarm')
        self.plot_matrix(dequantized_tensor, axes[2], 'Dequantized Tensor', cmap='coolwarm')
        q_error_tensor = abs(original_tensor - dequantized_tensor)
        self.plot_matrix(q_error_tensor, axes[3], 'Quantization Error Tensor', cmap=ListedColormap(['white']))

        fig.tight_layout()
        plt.show()


