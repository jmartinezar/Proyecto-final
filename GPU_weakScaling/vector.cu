#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel to perform vector addition
__global__ void vectorAdd(const double *A, const double *B, double *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[]) {
    int base_size = std::atoi(argv[1]);  // Tamaño base del vector
    int scale_factor = std::atoi(argv[2]); // Factor de escalamiento

    int n = base_size * scale_factor;
    size_t size = n * sizeof(double);

    // Allocate memory for host vectors
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i * 2.0f;
    }

    // Allocate memory for device vectors
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::system_clock::now(); //start time
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now(); //end time

    std::chrono::duration<double> elapsed_seconds = end-start;

    // Total time
    double wtime = elapsed_seconds.count();

    std::cout << scale_factor << "\t" << wtime << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
