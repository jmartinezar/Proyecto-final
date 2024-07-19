#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA kernel for matrix multiplication
__global__ void matrixMul(const double *A, const double *B, double *C, int width) {
    // Calculate row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the element of the product matrix
    if (row < width && col < width) {
        double value = 0;
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

int main(void) {
    // Size of the matrices (assuming square matrices for simplicity)
    int width = 1000000;
    size_t size = width * width * sizeof(double);

    // Allocate memory for host matrices
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C = (double *)malloc(size);

    // Initialize host matrices
    for (long int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate memory for device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy host matrices to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    // Launch the matrix multiplication kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print a few elements of the result matrix
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i*width+1, h_C[i*width+1]);
    }

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
