#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

void hell0_world() {
    printf("hello world\n");
}

__global__ void hell0_world_hernel() {
    printf("hello world from kernel: threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d, blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d \n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);

}

int main(int args, char* argv[]) {
    hell0_world();
    hell0_world_hernel<<<10,10, 10>>>();
    cudaDeviceReset();
    return 0;
}