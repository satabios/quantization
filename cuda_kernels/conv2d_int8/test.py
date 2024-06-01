import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
import numpy as np

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6'
os.environ['MAX_JOBS'] = '12'

# Load CUDA kernel
conv2d_cuda = load(name='conv2d', sources=['conv2d_w8a8.cu'])

# Define input parameters
batch_size = 16
channel_in = 3
width = 32
height = 32
channel_out = 8
kernel_width = 3
kernel_height = 3

# Create random input and kernel
input_tensor = torch.rand(batch_size, channel_in, width, height, device='cuda')
kernel_tensor = torch.rand(channel_out, channel_in, kernel_width, kernel_height, device='cuda')

# Compute output dimensions
out_width = width - kernel_width + 1
out_height = height - kernel_height + 1

# Allocate output tensor
output_tensor = torch.zeros(batch_size, channel_out, out_height, out_width, device='cuda')

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Run CUDA kernel
start_event.record()
conv2d_cuda.conv2d(input_tensor, kernel_tensor, output_tensor, 
                   batch_size, channel_in, width, height, 
                   channel_out, kernel_width, kernel_height)
end_event.record()

# Wait for the events to be recorded
torch.cuda.synchronize()
custom_conv_time = start_event.elapsed_time(end_event)

# PyTorch's conv2d for comparison
input_tensor_pt = input_tensor.detach().clone()
kernel_tensor_pt = kernel_tensor.detach().clone()

start_event.record()
output_tensor_pt = F.conv2d(input_tensor_pt, kernel_tensor_pt)
end_event.record()

# Wait for the events to be recorded
torch.cuda.synchronize()
torch_conv_time = start_event.elapsed_time(end_event)

# Compare the outputs
print('CUDA Output:', output_tensor)
print('PyTorch Output:', output_tensor_pt)
print('Outputs match:', torch.allclose(output_tensor, output_tensor_pt))


print(f"Custom 2D convolution time: {custom_conv_time:.3f} ms")
print(f"PyTorch F.conv2d time: {torch_conv_time:.3f} ms")