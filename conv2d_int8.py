import torch


def im2col(input, kernel_size, stride=1, padding=0):
    batch_size, channels, height, width = input.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Apply padding
    input_padded = torch.nn.functional.pad(input, (padding, padding, padding, padding))

    # Extract patches
    patches = input_padded.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, out_height * out_width, -1)

    return patches


def pad_to_multiple_of_8(tensor):
    size = tensor.size(-1)
    pad_size = (8 - (size % 8)) % 8  # Calculate the amount of padding needed
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))
    return tensor


def conv2d_int8(input, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Step 1: Extract patches
    patches = im2col(input, kernel_size, stride,
                     padding)  # Shape: [batch_size, out_height * out_width, in_channels * kernel_size * kernel_size]

    # Step 2: Pad patches to have columns as a multiple of 8
    patches_padded = pad_to_multiple_of_8(patches.view(-1, patches.shape[-1]))

    # Step 3: Perform matrix multiplication using torch._int_mm
    weight_reshaped = weight.view(out_channels,
                                  -1).t()  # Shape: [in_channels * kernel_size * kernel_size, out_channels]
    weight_padded = pad_to_multiple_of_8(weight_reshaped)

    output_patches = torch._int_mm(patches_padded,
                                   weight_padded)  # Shape: [batch_size * out_height * out_width, out_channels]
    output_patches = output_patches.view(batch_size, -1,
                                         out_channels)  # Shape: [batch_size, out_height * out_width, out_channels]

    # Step 4: Reshape the result back to the output tensor shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    output = output_patches.permute(0, 2, 1).contiguous().view(batch_size, out_channels, out_height, out_width)

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

# Example usage
input = torch.randn(1, 3, 5, 5)  # Example input tensor
weight = torch.randn(2, 3, 3, 3)  # Example weight tensor (2 filters, 3 input channels, 3x3 kernel)
bias = torch.randn(2)  # Example bias tensor

output = conv2d_int8(input.to('cuda'), weight.to('cuda'), bias.to('cuda'), stride=1, padding=1)
print(output)
