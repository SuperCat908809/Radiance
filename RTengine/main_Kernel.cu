#include "main_Kernel.h"

#include <iostream>
#include <stb/stb_image_write.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int width, int height, uint8_t* image) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid >= width * height) return;

	int y = gid / width;
	int x = gid % width;

	float r = x / (width - 1.0f);
	float g = y / (height - 1.0f);
	float b = (r + g) / 2.0f;

	image[gid * 3 + 0] = static_cast<uint8_t>(r * 255.999f);
	image[gid * 3 + 1] = static_cast<uint8_t>(g * 255.999f);
	image[gid * 3 + 2] = static_cast<uint8_t>(b * 255.999f);
}

void Kernel::Run() {

	std::cout << "Starting kernel.\n";

	uint8_t* d_image{ nullptr };
	std::cout << "Allocating image memory on device.\n";
	cudaMalloc((void**)&d_image, width * height * 3 * sizeof(uint8_t));


	int threads = 32;
	int blocks = (width * height + threads - 1) / threads;

	std::cout << "Launching render kernel.\n";
	kernel<<<blocks, threads>>>(width, height, d_image);
	cudaDeviceSynchronize();

	std::cout << "Downloading rendered image from device to host.\n";
	uint8_t* h_image = new uint8_t[width * height * 3];
	cudaMemcpy(h_image, d_image, width * height * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(d_image);


	std::cout << "Writing image to disk.\n";
	stbi_flip_vertically_on_write(true);
	stbi_write_jpg("kernel_testing.jpg", width, height, 3, h_image, 90);


	delete[] h_image;


	std::cout << "Kernel operation finished.\n";
}