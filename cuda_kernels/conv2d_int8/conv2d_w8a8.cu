#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolution2DKernel(const int8_t* input, const int8_t* kernel, int32_t* output,
                                    const int inputWidth, const int inputHeight,
                                    const int kernelWidth, const int kernelHeight,
                                    const int stride, const int padding) {
    extern __shared__ int8_t sharedMemory[];

    // Calculate output dimensions
    const int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
    const int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;

    // 2D indices for the current thread within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global indices for the current thread
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    // Calculate shared memory dimensions
    const int sharedInputWidth = blockDim.x + kernelWidth - 1;
    const int sharedInputHeight = blockDim.y + kernelHeight - 1;

    // Pointers to shared memory
    int8_t* sharedInput = sharedMemory;
    int8_t* sharedKernel = sharedMemory + sharedInputWidth * sharedInputHeight;

    // Load input elements into shared memory with loop tiling
    for (int i = ty; i < sharedInputHeight; i += blockDim.y) {
        for (int j = tx; j < sharedInputWidth; j += blockDim.x) {
            int inputX = blockIdx.x * blockDim.x + j - padding;
            int inputY = blockIdx.y * blockDim.y + i - padding;

            if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
                sharedInput[i * sharedInputWidth + j] = input[inputY * inputWidth + inputX];
            } else {
                sharedInput[i * sharedInputWidth + j] = 0;
            }
        }
    }

    // Load kernel elements into shared memory
    for (int i = ty; i < kernelHeight; i += blockDim.y) {
        for (int j = tx; j < kernelWidth; j += blockDim.x) {
            sharedKernel[i * kernelWidth + j] = kernel[i * kernelWidth + j];
        }
    }

    __syncthreads();

    if (x < outputWidth && y < outputHeight) {
        int32_t sum = 0;

        // Perform convolution with loop unrolling and reordering
        for (int i = 0; i < kernelHeight; i++) {
            #pragma unroll
            for (int j = 0; j < kernelWidth; j++) {
                int inputX = tx + j;
                int inputY = ty + i;

                sum += static_cast<int32_t>(sharedInput[inputY * sharedInputWidth + inputX]) *
                       static_cast<int32_t>(sharedKernel[i * kernelWidth + j]);
            }
        }

        output[y * outputWidth + x] = sum;
    }
}



void conv2d(const torch::Tensor& input, const torch::Tensor& kernel, torch::Tensor& output, int stride, int padding) {
    const int inputWidth = input.size(1);
    const int inputHeight = input.size(0);
    const int kernelWidth = kernel.size(1);
    const int kernelHeight = kernel.size(0);

    const int outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
    const int outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;

    output.resize_({outputHeight, outputWidth});

    const dim3 blockSize(32, 32);
    const dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);
    size_t sharedMemorySize = (blockSize.x + kernelWidth - 1) * (blockSize.y + kernelHeight - 1) * sizeof(int8_t) +
                              kernelWidth * kernelHeight * sizeof(int8_t);

    convolution2DKernel<<<gridSize, blockSize, sharedMemorySize>>>(
        input.data_ptr<int8_t>(),
        kernel.data_ptr<int8_t>(),
        output.data_ptr<int32_t>(),
        inputWidth, inputHeight,
        kernelWidth, kernelHeight,
        stride, padding
    );

    cudaDeviceSynchronize();
}

// Python binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "2D Convolution with CUDA");
}
