import os
# os.environ['MAX_JOBS'] = '12'
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import torch
from torch.utils.cpp_extension import load

# Load the CUDA kernel
conv2d_w8a8 = load(
    name="conv2d_w8a8",
    sources=["conv2d_w8a8.cu"],
    verbose=True
)

class Conv2dW8A8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, padding, stride):
        output_height = (input.size(2) + 2 * padding - weights.size(2)) // stride + 1
        output_width = (input.size(3) + 2 * padding - weights.size(3)) // stride + 1
        output = torch.zeros(
            (input.size(0), weights.size(0), output_height, output_width),
            dtype=torch.int32, device=input.device
        )
        conv2d_w8a8.conv2d_w8a8(input, weights, output, padding, stride)
        return output.float()  # Convert output to float32

# Usage example
torch.manual_seed(1250)
inputs = torch.randint(-128, 127, (1, 3, 5, 5), dtype=torch.int8, device='cuda')
weights = torch.randint(-128, 127, (4, 3, 3, 3), dtype=torch.int8, device='cuda')

output = Conv2dW8A8.apply(inputs, weights, 0, 1)
output_float = torch.nn.functional.conv2d(inputs.float(), weights.float(), padding=0, stride=1)  # Cast inputs to float32

print("Input: \n", inputs)
print("Weights:\n", weights)
print("PyTorch Output:\n", output_float)
print("CUDA Output:\n", output.T)

for i in range(output.shape[1]):
    print(torch.allclose(output[0, i], output_float[0, i], atol=1e-4))


