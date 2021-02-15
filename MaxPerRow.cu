
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

// CUDA kernel. Each thread takes care of one element of c


__global__ void vecAdd(double* a, double* c, int width, int height) {

	int id = threadIdx.x;
	int k = 0;
	double max = 0;
	double value;
	// Make sure we do not go out of bounds
	for (k = 0; k < width; k++) {
		value = a[id * width + k];
		if (max <= value)
			max = value;
		c[id] = max;
	}
}


int main(int argc, char* argv[])
{
	// Size of vectors should less than 1024

	int width = 5;
	int height = 10;
	int n = width * height;
	// Host input vectors
	double* h_a;
	//Host output vector
	double* h_c;

	// Device input vectors
	double* d_a;
	//Device output vector
	double* d_c;

	// Size, in bytes, of each vector
	size_t bytes = n * sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double*)malloc(bytes);
	h_c = (double*)malloc(height * sizeof(double));

	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_c, height * sizeof(double));

	int i;
	// Initialize vectors on host
	for (i = 0; i < n; i++) {
		h_a[i] = i;

	}
	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

	int blockSize;

	// Number of threads in each thread block
	blockSize = height;

	// Number of thread blocks in grid
	dim3 gridSize(1, 1);

	vecAdd << <gridSize, blockSize >> > (d_a, d_c, width, height);

	// Copy array back to host
	cudaMemcpy(h_c, d_c, height * sizeof(double), cudaMemcpyDeviceToHost);

	// Sum up vector c and print result divided by n, this should equal 1 within error

	for (int i = 0; i < height; i++) {	
		printf("%f  ", h_c[i]);
		printf("\n");
	}


	// Release device memory
	cudaFree(d_a);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_c);

	return 0;
}  
