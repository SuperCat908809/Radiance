#include "main_Kernel.h"

#include <iostream>

#include <stb/stb_image_write.h>
#include <glm/glm.hpp>
#include <easylogging/easylogging++.h>

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
	LOG(INFO) << "Allocating image memory on device";
	cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3));
	LOG(INFO) << "allocation finished";
}

void Renderer_cu::Run() {

	int threads = 32;
	int blocks = (width * height + threads - 1) / threads;

	LOG(INFO) << "Launching render kernel";
	kernel<<<blocks, threads>>>(width, height, d_image);
	cudaDeviceSynchronize();
	LOG(INFO) << "kernel finished";
}

std::vector<glm::vec3> Renderer_cu::Download() {

	LOG(INFO) << "Downloading kernel image from device";
	std::vector<glm::vec3> h_image(width * height, glm::vec3(0.0f));
	cudaMemcpy((glm::vec3*)h_image.data(), d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	LOG(INFO) << "download done";

	return h_image;
}

void Renderer_cu::Delete() {

	LOG(INFO) << "Deleting kernel device memory";
	if (d_image == nullptr) LOG(WARNING) << "Attempting to free after free device memory";

	cudaFree(d_image);
	d_image = nullptr;

	LOG(INFO) << "deletion finished";
}