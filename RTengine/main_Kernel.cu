#include "main_Kernel.h"

#include <iostream>

#include <stb/stb_image_write.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int width, int height, glm::vec3* image) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;
	if (gid >= width * height) return;

	int y = gid / width;
	int x = gid % width;

	float r = x / (width - 1.0f);
	float g = y / (height - 1.0f);
	float b = (r + g) / 2.0f;

	image[gid][0] = r;
	image[gid][1] = g;
	image[gid][2] = b;
}

Renderer_cu::Renderer_cu(int width, int height) : width(width), height(height) {
	std::cout << "Allocating image memory on device... ";
	cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3));
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

std::vector<glm::vec3> Renderer_cu::Download() {
	std::cout << "Downloading kernel image from device... ";
	std::vector<glm::vec3> h_image(width * height, glm::vec3(0.0f));
	cudaMemcpy((glm::vec3*)h_image.data(), d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	std::cout << "download done.\n";
	return h_image;
}

void Renderer_cu::Delete() {
	std::cout << "Deleting kernel device memory... ";
	cudaFree(d_image);
	d_image = nullptr;
	std::cout << "deletion finished.\n";
}