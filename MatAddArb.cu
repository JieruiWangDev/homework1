
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;


const int M = 1024;
const int N = 1024;


// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double a[M][N], double b[M][N], double c[M][N])
{
	// Get our global thread ID
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < M && j < N)
	{
		c[i][j] = a[i][j] + b[i][j];
	}

}


int main(int argc, char* argv[])
{
	// Size of vectors
	

	// Host input vectors
	double(*h_a)[N] = new double[M][N];
	double(*h_b)[N] = new double[M][N];

	//Host output vector
	double(*h_c)[N] = new double[M][N];

	// Device input vectors
	double(*d_a)[N];
	double(*d_b)[N];
	
	//Device output vector
	double(*d_c)[N];

	// Size, in bytes, of each vector
	size_t bytes = sizeof(double) * M * N;

	// Allocate memory for each vector on host
	//h_a = (double*)malloc(bytes);
	//h_b = (double*)malloc(bytes);
	//h_c = (double*)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc((void**)&d_a, bytes);
	cudaMalloc((void**)&d_b, bytes);
	cudaMalloc((void**)&d_c, bytes);

	int i;
	// Initialize vectors on host
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			h_a[i][j] = sin(i) * sin(i);
			h_b[i][j] = cos(i) * cos(i);
		}
	}

	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// Number of threads in each thread block
	dim3 blockSize(32, 32);

		// Number of thread blocks in grid
	dim3 gridSize(1, 1);

	vecAdd << <gridSize, blockSize >> > (d_a, d_b, d_c);

	// Copy array back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Sum up vector c and print result divided by n, this should equal 1 within error

	for (int j = 0; j < 5; j++) {
		for (i = 0; i < 5; i++) {
			printf("%f  ", h_c[i][j]);
		}
		printf("\n");
	}


	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}