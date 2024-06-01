#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void conv2d_kernel(float *input, float *kernel, float *output, 
                              int batch_size, int channel_in, int width, int height, 
                              int channel_out, int kernel_width, int kernel_height) {
    int batch_idx = blockIdx.z;
    int out_c = blockIdx.y;
    int in_c = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    int out_width = width - kernel_width + 1;
    int out_height = height - kernel_height + 1;

    if (row < out_height && col < out_width) {
        float value = 0.0;

        for (int k_row = 0; k_row < kernel_height; ++k_row) {
            for (int k_col = 0; k_col < kernel_width; ++k_col) {
                int input_row = row + k_row;
                int input_col = col + k_col;
                int input_idx = ((batch_idx * channel_in + in_c) * height + input_row) * width + input_col;
                int kernel_idx = ((out_c * channel_in + in_c) * kernel_height + k_row) * kernel_width + k_col;
                value += input[input_idx] * kernel[kernel_idx];
            }
        }

        int output_idx = ((batch_idx * channel_out + out_c) * out_height + row) * out_width + col;
        output[output_idx] = value;
    }
}

void conv2d(torch::Tensor input, torch::Tensor kernel, torch::Tensor output, 
            int batch_size, int channel_in, int width, int height, 
            int channel_out, int kernel_width, int kernel_height) {
    dim3 threads(32, 32);
    dim3 blocks(channel_in, channel_out, batch_size);

    conv2d_kernel<<<blocks, threads>>>(input.data_ptr<float>(), kernel.data_ptr<float>(), output.data_ptr<float>(), 
                                       batch_size, channel_in, width, height, 
                                       channel_out, kernel_width, kernel_height);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "2D Convolution");
}
