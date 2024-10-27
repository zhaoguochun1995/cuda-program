#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include "utils.h"
//#include <stdio.h>

template <typename T>
void printMatrix(const T* ptr, size_t rows, size_t cols) {
    for (size_t row_index = 0; row_index < rows; ++ row_index) {
        const T* row_ptr = ptr + (row_index * cols);
        for (size_t col_index = 0; col_index < cols; ++ col_index) {
            std::cout << " " << row_ptr[col_index] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
void hostMatrixAdd(const T* A, const T* B, T* C, size_t rows, size_t cols) {
    for (size_t row_index = 0; row_index < rows; ++ row_index) {
        for (size_t col_index = 0; col_index < cols; ++ col_index) {
            size_t bias = row_index * cols + col_index;
            *(C + bias) = *(A + bias) + *(B + bias);
        }
    }
}

__global__ void  print_kernerl_idx() {
    printf("blockDim.x: %d, blockDim.y: %d, blockDim.z:%d, gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    printf("kernel: threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
}

template <typename T>
__global__ void deviceMatrixAddGrid2dBlock2d(const T* A, const T* B, T* C, size_t rows, size_t cols) {
    int col_index = threadIdx.x + blockIdx.x * blockDim.x;
    int row_index = threadIdx.y + blockIdx.y * blockDim.y;
    int index = row_index * cols + col_index;
    // printf("index:%d\t", index);
    if (index < (rows * cols)) {
       *(C + index) = *(A + index) + *(B + index);
    }
}


int main(int argc, char** argv) {
    using T = float;
    const int ROWS = 8;
    const int COLS = 16;
    std::vector<T> hostMatrixA(ROWS * COLS);
    std::vector<T> hostMatrixB(ROWS * COLS);
    std::vector<T> hostMatrixC(ROWS * COLS);
    std::vector<T> hostMatrixDeviceRef(ROWS * COLS);
    for (size_t i = 0; i < hostMatrixA.size(); ++i) {
        hostMatrixA[i] = i + 100;
        hostMatrixB[i] = 2 * i + 100;
    }
    hostMatrixAdd(hostMatrixA.data(), hostMatrixB.data(), hostMatrixC.data(), ROWS, COLS);

    printMatrix(hostMatrixA.data(), ROWS, COLS);
    printMatrix(hostMatrixB.data(), ROWS, COLS);
    printMatrix(hostMatrixC.data(), ROWS, COLS);

    T* deviceMatrixA = nullptr;
    T* deviceMatrixB = nullptr;
    T* deviceMatrixC = nullptr;
    CALL_CUDA(cudaMalloc(&deviceMatrixA, sizeof(T) * ROWS * COLS));
    CALL_CUDA(cudaMalloc(&deviceMatrixB, sizeof(T) * ROWS * COLS));
    CALL_CUDA(cudaMalloc(&deviceMatrixC, sizeof(T) * ROWS * COLS));

    CALL_CUDA(cudaMemcpy(deviceMatrixA, hostMatrixA.data(), sizeof(T) * hostMatrixA.size(), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMemcpy(deviceMatrixB, hostMatrixB.data(), sizeof(T) * hostMatrixB.size(), cudaMemcpyHostToDevice));


    {
        CALL_CUDA(cudaMemset(deviceMatrixC, __LINE__ - 1, sizeof(T) * COLS * ROWS));
        std::fill(hostMatrixDeviceRef.begin(), hostMatrixDeviceRef.end(), __LINE__);
        const int block_size_x = 32;
        const int block_size_y = block_size_x;
        const dim3 block_size(block_size_x, block_size_y);
        const int grid_size_x = (ROWS + block_size_x - 1) / block_size_x;
        const int grid_size_y = (COLS + block_size_y - 1) / block_size_y;
        const dim3 grid_size(grid_size_x, grid_size_y);
        print_kernerl_idx<<<grid_size,  block_size>>>();
        std::cout << "deviceMatrixAddGrid2dBlock2d<<<(" << grid_size.x << "," << grid_size.y << "), (" << block_size.x << "," << block_size.y << ") >>>" << std::endl; 
        deviceMatrixAddGrid2dBlock2d<<<grid_size, block_size>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, ROWS, COLS);
        CALL_CUDA(cudaDeviceSynchronize());
        CALL_CUDA(cudaMemcpy(hostMatrixDeviceRef.data(), deviceMatrixC, sizeof(T) * ROWS * COLS, cudaMemcpyDeviceToHost));
        printMatrix(hostMatrixDeviceRef.data(), ROWS, COLS);
        auto max_diff = checkResult(hostMatrixC, hostMatrixDeviceRef);
        std::cout << "max_diff:" << max_diff << std::endl;
        assert(max_diff == 0);
    }

    CALL_CUDA(cudaFree(deviceMatrixA));
    CALL_CUDA(cudaFree(deviceMatrixB));
    CALL_CUDA(cudaFree(deviceMatrixC));
    cudaDeviceReset();
}
