#include "main_Kernel.h"

#include <iostream>
#include <cassert>

#include <stb/stb_image_write.h>
#include <glm/glm.hpp>
#include <easylogging/easylogging++.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void kernel(int width, int height, glm::vec3* image) {
	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (gidx >= width || gidy >= height) return;

	int gid = gidy * width + gidx;

	int y = gidy;
	int x = gidx;

	float r = x / (width - 1.0f);
	float g = y / (height - 1.0f);
	float b = (r + g) / 2.0f;

	image[gid][0] = r;
	image[gid][1] = g;
	image[gid][2] = b;
}

Renderer_cu::Renderer_cu(int width, int height) : width(width), height(height) {
	assert(width > 0 && height > 0);

	int kb_allocated = width * height * sizeof(glm::vec3) / 1000;
	LOG(INFO) << "Renderer_cu::Renderer_cu ==> Allocating " << kb_allocated << "KB for a " << width << "x" << height << " image on device";
	cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3));
	LOG(INFO) << "Renderer_cu::Renderer_cu ==> allocation finished";
}

void Renderer_cu::Run() {

	dim3 threads = { 8,8,1 };
	//int blocks = (width * height + threads - 1) / threads;
	dim3 blocks{};
	blocks.x = (width + threads.x - 1) / threads.x;
	blocks.y = (height + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer_cu::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y;
	kernel<<<blocks, threads>>>(width, height, d_image);
	cudaDeviceSynchronize();
	LOG(INFO) << "Renderer_cu::Run ==> kernel finished";
}

std::vector<glm::vec3> Renderer_cu::Download() {

	LOG(INFO) << "Renderer_cu::Download ==> Downloading kernel image from device";
	std::vector<glm::vec3> h_image(width * height, glm::vec3(0.0f));
	cudaMemcpy((glm::vec3*)h_image.data(), d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	LOG(INFO) << "Renderer_cu::Download ==> download done";

	return h_image;
}

void Renderer_cu::Delete() {

	LOG(INFO) << "Renderer_cu::Delete ==> Deleting kernel device memory";
	if (d_image == nullptr) LOG(WARNING) << "Renderer_cu::Delete ==> Attempting to free after free device memory";

	cudaFree(d_image);
	d_image = nullptr;

	LOG(INFO) << "Renderer_cu::Delete ==> deletion finished";
}