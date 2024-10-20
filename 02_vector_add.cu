#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

using namespace std::chrono;


#define CALL_CUDA(expr) \
    do {    \
        cudaError_t code = expr; \
        if (code != cudaSuccess)  { \
            std::cout << __FILE__ << ":" << __LINE__ << cudaGetErrorString(code) << std::endl; \
        } \
    } while(0);

template <typename T>
void vectorAddCPU(const T* a, const T* b, T* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }
} 


template <typename T>
__global__ void vectorAddCUDASerial(const T* a, const T* b, T* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[i] = a[i] + b[i];
    }
} 


template <typename T>
__global__ void vectorAddCUDAParallel(const T* a, const T* b, T* out, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
} 

template <typename T>
T checkResult(const std::vector<T>& cpu, const std::vector<T>& gpu, bool print_detail = false) {
    T max_diff = 0;
    size_t N = cpu.size();
    for (size_t i = 0; i < N; i++) {
        T diff = std::abs(cpu[i] - gpu[i]);
        if (print_detail) {
            std::cout << cpu[i] << "\t" << gpu[i] << "\t" << diff << std::endl;
        }
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

int main(int argc, char** argv) {
    using T = float;
    const int N = 64 << 20;
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

    {
        auto start = std::chrono::high_resolution_clock::now();
        vectorAddCPU(a_h.data(), b_h.data(), c_h.data(), N);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "vectorAddCPU took " << duration.count() << " milliseconds to execute." << std::endl;
    }

    {
        int block_size = 1;
        int grid_size = 1;
        CALL_CUDA(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        vectorAddCUDASerial<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
        CALL_CUDA(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "grid_size:" << grid_size << " block_size:" << block_size << " ";
        std::cout << "vectorAddCUDASerial took " << duration.count() << " milliseconds to execute." << std::endl;

        std::vector<T> c_d_ref(N);
        CALL_CUDA(cudaMemcpy(c_d_ref.data(), c_d, sizeof(T) * N, cudaMemcpyDeviceToHost));

        T max_diff = checkResult(c_h, c_d_ref);
        std::cout << "cpu and gpu max diff:" << max_diff << std::endl;
    }
    std::cout << "grid_size,block_size,elasped_time,max_diff" << std::endl;
    {
        for (int block_size = 1; block_size < 1025; ++block_size) {
            int grid_size = (N - block_size + 1) / block_size;
            CALL_CUDA(cudaDeviceSynchronize());
            auto start = std::chrono::high_resolution_clock::now();
            vectorAddCUDAParallel<<<grid_size, block_size>>>(a_d, b_d, c_d, N);
            CALL_CUDA(cudaDeviceSynchronize());
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::vector<T> c_d_ref(N);
            CALL_CUDA(cudaMemcpy(c_d_ref.data(), c_d, sizeof(T) * N, cudaMemcpyDeviceToHost));

            T max_diff = checkResult(c_h, c_d_ref);
            std::cout << grid_size << "," << block_size << "," << duration.count() << "," << max_diff << std::endl;
        }
    }   

    CALL_CUDA(cudaDeviceSynchronize());

    CALL_CUDA(cudaFree(a_d));
    CALL_CUDA(cudaFree(b_d));
    CALL_CUDA(cudaFree(c_d));
    CALL_CUDA(cudaDeviceReset());
}