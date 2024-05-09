import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

class visualizer:

    def plot_matrix(self, tensor, ax, title, vmin=0, vmax=1, cmap=None):
        sns.heatmap(tensor.cpu().numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".2f", cbar=False)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    def plot_quantization_errors(self, original_tensor, quantized_tensor, dequantized_tensor, dtype):
        n_bits = 8
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        self.plot_matrix(original_tensor, axes[0], 'Original Tensor', cmap=ListedColormap(['white']))
        q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        self.plot_matrix(quantized_tensor, axes[1], f'{n_bits}-bit Linear Quantized Tensor', vmin=q_min, vmax=q_max,
                         cmap='coolwarm')
        self.plot_matrix(dequantized_tensor, axes[2], 'Dequantized Tensor', cmap='coolwarm')
        q_error_tensor = abs(original_tensor - dequantized_tensor)
        self.plot_matrix(q_error_tensor, axes[3], 'Quantization Error Tensor', cmap=ListedColormap(['white']))

        fig.tight_layout()
        plt.show()


