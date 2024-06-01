import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Compile and load the custom 2D convolution module
conv2d_module = load(
    name="conv2d",
    sources=["conv2d_w8a8.cu"],  # Adjust the path to your CUDA source file if necessary
    verbose=True
)

# Define input and kernel tensors
dtype = torch.int8

input_tensor = torch.tensor([
    [1, 2, 3, 4, 6],
    [5, 6, 7, 8, 7],
    [9, 10, 11, 12, 1],
    [13, 14, 15, 16, 4],
    [13, 14, 15, 16, 4]
], dtype=dtype).cuda()

kernel_tensor = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=dtype).cuda()

# Define stride and padding
stride = 1
padding = 1

# Calculate the output dimensions
output_height = (input_tensor.shape[0] - kernel_tensor.shape[0] + 2 * padding) // stride + 1
output_width = (input_tensor.shape[1] - kernel_tensor.shape[1] + 2 * padding) // stride + 1

# Prepare an output tensor for the custom convolution
output_tensor_custom = torch.empty((output_height, output_width), dtype=torch.int32).cuda()

# Call the custom 2D convolution function
conv2d_module.conv2d(input_tensor, kernel_tensor, output_tensor_custom, stride, padding)

# Prepare the tensors for PyTorch's F.conv2d
input_tensor_4d = input_tensor.unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, height, width]
kernel_tensor_4d = kernel_tensor.unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, kernel_height, kernel_width]

# Call PyTorch's built-in F.conv2d function
output_tensor_torch = F.conv2d(input_tensor_4d, kernel_tensor_4d, stride=stride, padding=padding)

# Squeeze the output to get back to 2D shape
output_tensor_torch = output_tensor_torch.squeeze(0).squeeze(0)

# Print the results
print("Input Tensor:")
print(input_tensor.cpu())
print("Kernel Tensor:")
print(kernel_tensor.cpu())
print("Output Tensor (Custom):")
print(output_tensor_custom.cpu().float())
print("Output Tensor (PyTorch F.conv2d):")
print(output_tensor_torch.cpu())

# Compare the results
comparison = torch.allclose(output_tensor_custom.cpu().float(), output_tensor_torch.cpu(), atol=1e-6)
print("Do the results match?", comparison)
