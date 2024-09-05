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
    // Size of vectors
    // int n = 1000000;
    int n = std::atoi(argv[1]);
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

    // Number of threads per block
    int threadsPerBlock = 256;

    // Number of blocks in the grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector addition
    auto start = std::chrono::system_clock::now(); //start time
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now(); //end time

    std::chrono::duration<double> elapsed_seconds = end-start;
    // Total time
    double wtime = elapsed_seconds.count();
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a few elements of the result vector
    for (int i = 0; i < 10; i++) {
      fprintf(stderr,"%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    // Prints size and elapsed time in vector addition
    std::cout << size << "\t" << wtime << std::endl;
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
