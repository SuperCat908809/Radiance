#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void run_test();

__global__ void test_kernel() {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Hello world from t:%i b:%i g:%i\n", tid, bid, gid);
}

void run_test() {

	std::cout << "CUDA start.\n";

	int blocks = 2;
	int threads = 4;

	test_kernel<<<blocks, threads>>>();
	cudaDeviceSynchronize();

	cudaDeviceReset();

	std::cout << "CUDA finished.\n";

}