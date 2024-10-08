#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void matrixMul(const double *A, const double *B, double *C, int width_A, int high_A, int width_B, int high_B)
{
	if(width_A != high_B)
	{
		printf("The size of matrices isn't correct\n");
		return;
	}

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < high_A && col < width_B)
	{
		double value = 0;
		for (int k = 0; k < width_A; ++k)
		{
			value += A[row * width_A + k] * B[k * width_B + col];
        	}
		C[row * width_B + col] = value;
	}
}

int main(int argc, char *argv[])
{  
    if (argc < 3) {
      std::cerr << "Uso de: " << argv[0] << " <matrix_size> <number_of_threads>" << std::endl;
      return 1;
    }

    int size = std::atoi(argv[1]);
    int numberOfThreads = std::atoi(argv[2]);
    
    // TODO: rename high -> height
    int width_A = size;
    int high_A = size;
    int width_B = size;
    int high_B = size;

    size_t size_A = width_A * high_A * sizeof(double);
    size_t size_B = width_B * high_B * sizeof(double);
    size_t size_C = width_B * high_A * sizeof(double);

    double *h_A = (double *)malloc(size_A);
    double *h_B = (double *)malloc(size_B);
    double *h_C = (double *)malloc(size_C);

    for (int i = 0; i < width_A * high_A; i++)
    {
        h_A[i] = 1.0;
    }

    for (int i = 0; i < width_B * high_B; i++)
    {
        h_B[i] = 2.0;
    }

    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(numberOfThreads, numberOfThreads);
    dim3 dimGrid((width_B + dimBlock.x - 1) / dimBlock.x, (high_A + dimBlock.y - 1) / dimBlock.y);

    auto start = std::chrono::system_clock::now(); //start time
    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width_A, high_A, width_B, high_B);
    cudaDeviceSynchronize();
    auto end = std::chrono::system_clock::now(); //end time

    std::chrono::duration<double> elapsed_seconds = end-start;

    // Total time
    double wtime = elapsed_seconds.count();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
    {
        fprintf(stderr,"C[%d] = %f\n", i*width_B+1, h_C[i*width_B+1]);
    }

    // Prints size and elapsed time in matrix multiplication
    std::cout << size << "\t" << wtime << std::endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
