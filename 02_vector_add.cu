#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

#define CALL_CUDA(expr) \
    do {    \
        cudaError_t code = expr; \
        if (code != cudaSuccess)  { \
        } \
    } while(0);

template <typename T>
void vectorAddCPU(const T* a, const T* b, T* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }
} 


template <typename T>
__global__ void vectorAddCUDA(const T* a, const T* b, T* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }
} 

int main(int argc, char** argv) {
    using T = float;
    const int N = 1024;
    std::vector<T> a_h(N);
    std::vector<T> b_h(N);
    std::vector<T> c_h(N);

    for (size_t i = 0; i < N; ++i) {
        a_h[i] = std::sin(i);
        b_h[i] = std::cos(i) * 2;
        c_h[0] = 0;
    }

    T* a_d = nullptr;
    T* b_d = nullptr;
    T* c_d = nullptr;
    CALL_CUDA(cudaMalloc(&a_d, sizeof(T) * N));
    CALL_CUDA(cudaMalloc(&b_d, sizeof(T) * N));
    CALL_CUDA(cudaMalloc(&c_d, sizeof(T) * N));

    CALL_CUDA(cudaMemcpy(a_d, a_h.data(), sizeof(T) * N, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMemcpy(b_d, b_h.data(), sizeof(T) * N, cudaMemcpyHostToDevice));

    vectorAddCPU(a_h.data(), b_h.data(), c_h.data(), N);

    //Dim blockDim(256, 1, 1);
    int block_size = 256;
    int grid_size = (N - block_size + 1) / block_size;
    vectorAddCUDA<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    CALL_CUDA(cudaDeviceSynchronize());

    std::vector<T> c_d_ref(N);
    CALL_CUDA(cudaMemcpy(c_d_ref.data(), c_d, sizeof(T) * N, cudaMemcpyDeviceToHost));
    
    T max_diff = 0;
    for (size_t i = 0; i < N; i++) {
        T diff = std::abs(c_h[i] - c_d_ref[i]);
        std::cout << c_h[i] << "\t" << c_d_ref[i] << "\t" << diff << std::endl;
        max_diff = std::max(max_diff, diff);
    }
    std::cout << "cpu and gpu max diff:" << max_diff << std::endl;


    CALL_CUDA(cudaFree(a_d));
    CALL_CUDA(cudaFree(b_d));
    CALL_CUDA(cudaFree(c_d));
    CALL_CUDA(cudaDeviceReset());
}