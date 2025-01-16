#include "main_Kernel.h"

#include <iostream>
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int width, int height, float* image) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid >= width * height) return;

	int y = gid / width;
	int x = gid % width;

	float r = x / (width - 1.0f);
	float g = y / (height - 1.0f);
	float b = (r + g) / 2.0f;

	image[gid * 3 + 0] = r;
	image[gid * 3 + 1] = g;
	image[gid * 3 + 2] = b;
}

Renderer_cu::Renderer_cu(int width, int height) : width(width), height(height) {
	std::cout << "Allocating image memory on device... ";
	cudaMalloc((void**)&d_image, width * height * 3 * sizeof(float));
	std::cout << "allocation finished.\n";
}

void Renderer_cu::Run() {

	int threads = 32;
	int blocks = (width * height + threads - 1) / threads;

	std::cout << "Launching render kernel... \n";

	kernel<<<blocks, threads>>>(width, height, d_image);
	cudaDeviceSynchronize();

	std::cout << "kernel finished.\n";
}

std::vector<float> Renderer_cu::Download() {
	std::cout << "Downloading kernel image from device... ";
	std::vector<float> h_image(width * height * 3, 0.0f);
	cudaMemcpy((float*)h_image.data(), d_image, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << "download done.\n";
	return h_image;
}

void Renderer_cu::Delete() {
	std::cout << "Deleting kernel device memory... ";
	cudaFree(d_image);
	d_image = nullptr;
	std::cout << "deletion finished.\n";
}