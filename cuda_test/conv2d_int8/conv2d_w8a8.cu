// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>

// #define TILE_WIDTH 16
// #define KERNEL_SIZE 3

// // __global__ void conv2d_w8a8_kernel(
// //     const int8_t* __restrict__ input,
// //     const int8_t* __restrict__ weights,
// //     int32_t* __restrict__ output,
// //     int input_channels,
// //     int output_channels,
// //     int input_height,
// //     int input_width,
// //     int output_height,
// //     int output_width,
// //     int padding,
// //     int stride
// // ) {
// //     __shared__ int8_t shared_input[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
// //     __shared__ int8_t shared_weights[KERNEL_SIZE][KERNEL_SIZE][TILE_WIDTH];

// //     int tx = threadIdx.x;
// //     int ty = threadIdx.y;
// //     int row_o = blockIdx.y * TILE_WIDTH + ty;
// //     int col_o = blockIdx.x * TILE_WIDTH + tx;
// //     int row_i = row_o * stride - padding;
// //     int col_i = col_o * stride - padding;

// //     for (int c = 0; c < input_channels; ++c) {
// //         if (row_i + ty < input_height && row_i + ty >= 0 &&
// //             col_i + tx < input_width && col_i + tx >= 0) {
// //             shared_input[ty][tx] = input[(c * input_height + (row_i + ty)) * input_width + (col_i + tx)];
// //         } else {
// //             shared_input[ty][tx] = 0;
// //         }
// //         __syncthreads();

// //         for (int k = 0; k < output_channels; ++k) {
// //             if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
// //                 shared_weights[ty][tx][k] = weights[((k * input_channels + c) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx];
// //             }
// //             __syncthreads();

// //             int output_value = 0;
// //             if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < output_height && col_o < output_width) {
// //                 for (int i = 0; i < KERNEL_SIZE; ++i) {
// //                     for (int j = 0; j < KERNEL_SIZE; ++j) {
// //                         output_value += shared_input[ty + i][tx + j] * shared_weights[i][j][k];
// //                     }
// //                 }
// //                 atomicAdd(&output[(k * output_height + row_o) * output_width + col_o], output_value);
// //             }
// //             __syncthreads();
// //         }
// //     }
// // }
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// #define TILE_WIDTH 2  // Adjust TILE_WIDTH for debugging purposes
// #define KERNEL_SIZE 3  // Assuming a 3x3 kernel

// __global__ void conv2d_w8a8_kernel(
//     const int8_t* __restrict__ input,
//     const int8_t* __restrict__ weights,
//     int32_t* __restrict__ output,
//     int input_channels,
//     int output_channels,
//     int input_height,
//     int input_width,
//     int output_height,
//     int output_width,
//     int padding,
//     int stride
// ) {
//     __shared__ int8_t shared_input[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
//     __shared__ int8_t shared_weights[KERNEL_SIZE][KERNEL_SIZE][TILE_WIDTH];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int row_o = blockIdx.y * TILE_WIDTH + ty;
//     int col_o = blockIdx.x * TILE_WIDTH + tx;
//     int row_i = row_o * stride - padding;
//     int col_i = col_o * stride - padding;

//     // Initialize the output value for each thread
//     int32_t output_value[TILE_WIDTH] = {0};

//     for (int c = 0; c < input_channels; ++c) {
//         // Load input into shared memory
//         for (int i = ty; i < TILE_WIDTH + KERNEL_SIZE - 1; i += blockDim.y) {
//             for (int j = tx; j < TILE_WIDTH + KERNEL_SIZE - 1; j += blockDim.x) {
//                 int in_row = row_i + i;
//                 int in_col = col_i + j;
//                 if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
//                     shared_input[i][j] = input[(c * input_height + in_row) * input_width + in_col];
//                 } else {
//                     shared_input[i][j] = 0;
//                 }
//             }
//         }

//         // Load weights into shared memory
//         if (ty < KERNEL_SIZE && tx < KERNEL_SIZE) {
//             for (int k = 0; k < output_channels; ++k) {
//                 shared_weights[ty][tx][k] = weights[((k * input_channels + c) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx];
//             }
//         }
//         __syncthreads();

//         // Perform convolution
//         for (int k = 0; k < output_channels; ++k) {
//             if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < output_height && col_o < output_width) {
//                 for (int i = 0; i < KERNEL_SIZE; ++i) {
//                     for (int j = 0; j < KERNEL_SIZE; ++j) {
//                         output_value[k] += shared_input[ty + i][tx + j] * shared_weights[i][j][k];
//                     }
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     // Write the output values
//     for (int k = 0; k < output_channels; ++k) {
//         if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < output_height && col_o < output_width) {
//             atomicAdd(&output[(k * output_height + row_o) * output_width + col_o], output_value[k]);
//         }
//     }
// }

// // To call this kernel, ensure proper kernel configuration matching your input dimensions and capabilities.


// void conv2d_w8a8(
//     torch::Tensor input,
//     torch::Tensor weights,
//     torch::Tensor output,
//     int padding,
//     int stride
// ) {
//     const int batch_size = input.size(0);
//     const int input_channels = input.size(1);
//     const int input_height = input.size(2);
//     const int input_width = input.size(3);
//     const int output_channels = weights.size(0);
//     const int output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
//     const int output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;

//     const dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
//     const dim3 gridDim((output_width + TILE_WIDTH - 1) / TILE_WIDTH, (output_height + TILE_WIDTH - 1) / TILE_WIDTH);


//     for (int n = 0; n < batch_size; ++n) {
//         conv2d_w8a8_kernel<<<gridDim, blockDim>>>(
//             input[n].data_ptr<int8_t>(),
//             weights.data_ptr<int8_t>(),
//             output[n].data_ptr<int32_t>(),
//             input_channels,
//             output_channels,
//             input_height,
//             input_width,
//             output_height,
//             output_width,
//             padding,
//             stride
//         );
//     }
// }

// // Define the module
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("conv2d_w8a8", &conv2d_w8a8, "Convolution 2D with 8-bit weights and activations with padding and stride");
// }

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_WIDTH 16
#define KERNEL_SIZE 3

__device__ void atomicAdd32(int32_t* address, int32_t value) {
    atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
}

#define TILE_WIDTH 16
#define KERNEL_SIZE 3

__global__ void conv2d_w8a8_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    int32_t* __restrict__ output,
    int input_channels,
    int output_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int padding,
    int stride
) {
    // Shared memory 
    __shared__ int8_t shared_input[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];
    __shared__ int8_t shared_weights[KERNEL_SIZE][KERNEL_SIZE][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o * stride - padding;
    int col_i = col_o * stride - padding;

    // Zero-initialize shared memory
    if (tx < TILE_WIDTH + KERNEL_SIZE - 1 && ty < TILE_WIDTH + KERNEL_SIZE - 1) {
        shared_input[ty][tx] = 0;
    }
    if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
        for (int k = 0; k < TILE_WIDTH; ++k) {
            shared_weights[ty][tx][k] = 0;
        }
    }
    __syncthreads();

    for (int c = 0; c < input_channels; ++c) {
        if (row_i + ty < input_height && row_i + ty >= 0 && col_i + tx < input_width && col_i + tx >= 0) {
            shared_input[ty][tx] = input[(c * input_height + (row_i + ty)) * input_width + (col_i + tx)];
        } else {
            shared_input[ty][tx] = 0;
        }
        __syncthreads();

        for (int k = 0; k < output_channels; ++k) {
            if (tx < KERNEL_SIZE && ty < KERNEL_SIZE) {
                shared_weights[ty][tx][k] = weights[((k * input_channels + c) * KERNEL_SIZE + ty) * KERNEL_SIZE + tx];
            }
            __syncthreads();

            if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < output_height && col_o < output_width) {
                int output_value = 0; // Reset output value for each output element
                for (int i = 0; i < KERNEL_SIZE; ++i) {
                    for (int j = 0; j < KERNEL_SIZE; ++j) {
                        output_value += shared_input[ty + i][tx + j] * shared_weights[i][j][k];
                    }
                }
                atomicAdd(&output[(k * output_height + row_o) * output_width + col_o], output_value);
            }
            __syncthreads();
        }
    }
}


void conv2d_w8a8(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    int padding,
    int stride
) {
    const int batch_size = input.size(0);
    const int input_channels = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_channels = weights.size(0);
    const int output_height = (input_height + 2 * padding - KERNEL_SIZE) / stride + 1;
    const int output_width = (input_width + 2 * padding - KERNEL_SIZE) / stride + 1;

    const dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    const dim3 gridDim((output_width + TILE_WIDTH - 1) / TILE_WIDTH, (output_height + TILE_WIDTH - 1) / TILE_WIDTH);

    for (int n = 0; n < batch_size; ++n) {
        conv2d_w8a8_kernel<<<gridDim, blockDim>>>(
            input[n].data_ptr<int8_t>(),
            weights.data_ptr<int8_t>(),
            output[n].data_ptr<int32_t>(),
            input_channels,
            output_channels,
            input_height,
            input_width,
            output_height,
            output_width,
            padding,
            stride
        );
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_w8a8", &conv2d_w8a8, "Convolution 2D with 8-bit weights and activations with padding and stride");
}
