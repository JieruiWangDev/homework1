#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double* a, double* b, double* c, int width)
{
	// Get our global thread ID
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// Make sure we do not go out of bounds
	if (id < width * width) {
		c[id] = a[id] + b[id];
	}

}

__global__ void vecAddByRow(double* a, double* b, double* c, int width) {

	int id = threadIdx.x;
	int k = 0;
	double value;
	// Make sure we do not go out of bounds
	if (id < width) {
		for (k = 0; k < width; k++) {
			value = a[id * width + k] + b[id * width + k];
			c[id * width + k] = value;
		}

	}


}

__global__ void vecAddByCol(double* a, double* b, double* c, int col) {

	int id = threadIdx.x;
	int k = 0;
	double value;
	// Make sure we do not go out of bounds
	if (id < col) {
		for (k = 0; k < col; k++) {
			value = a[k * col + id] + b[k * col + id];
			c[k * col + id] = value;
		}
	}
}


int main(int argc, char* argv[])
{
	// Size of vectors
	int n = 1024 * 1024;
	int width = 1024;

	// Host input vectors
	double* h_a;
	double* h_b;
	//Host output vector
	double* h_c;

	// Device input vectors
	double* d_a;
	double* d_b;
	//Device output vector
	double* d_c;

	// Size, in bytes, of each vector
	size_t bytes = n * sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	int i;
	// Initialize vectors on host
	for (i = 0; i < n; i++) {
		h_a[i] = sin(i) * sin(i);
		h_b[i] = cos(i) * cos(i);
	}


	for (int j = 0; j < 5; j++) {
		for (i = 0; i < 5; i++) {
			printf("%f  ", h_a[j * width + i]);
		}
		printf("\n");
	}
	printf("\n\n\n\n");
	for (int j = 0; j < 5; j++) {
		for (i = 0; i < 5; i++) {
			printf("%f  ", h_b[j * width + i]);
		}
		printf("\n");
	}

	printf("\n\n\n\n");
	// Copy host vectors to device
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	int blockSize, gridSize;

	// Number of threads in each thread block
	blockSize = 1024;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)n / blockSize);

	vecAddByCol << <gridSize, blockSize >> > (d_a, d_b, d_c, width);

	// Copy array back to host
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// Sum up vector c and print result divided by n, this should equal 1 within error
	double sum = 0;

	for (int j = 0; j < 5; j++) {
		for (i = 0; i < 5; i++) {
			printf("%f  ", h_c[j * width + i]);
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

