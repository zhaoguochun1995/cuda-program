#pragma once

#define CALL_CUDA(expr) \
    do {    \
        cudaError_t code = expr; \
        if (code != cudaSuccess)  { \
            std::cout << __FILE__ << ":" << __LINE__ << cudaGetErrorString(code) << std::endl; \
        } \
    } while(0);



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