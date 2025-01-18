#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include <glm/glm.hpp>
#include <stb/stb_image_write.h>
#include <easylogging/easylogging++.h>

#include "cuError.h"

#define RT_ENGINE_IMPLEMENTATION
#include "main_Kernel.h"
#include "ray.h"


namespace RT_ENGINE {

__global__ void kernel(int width, int height, glm::vec3* image) {
	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (gidx >= width || gidy >= height) return;

	int gid = gidy * width + gidx;

	float u = (gidx / (width - 1.0f)) * 2.0f - 1.0f;
	float v = (gidy / (height - 1.0f)) * 2.0f - 1.0f;

	ray r(glm::vec3(0, 0, -4), glm::vec3(0, 0, 1));

	r.d += u * glm::vec3(1, 0, 0) + v * glm::vec3(0, 1, 0);

	float t = glm::normalize(r.d).y * 0.5f + 0.5f;
	glm::vec3 c = (1 - t) * glm::vec3(0.1f, 0.2f, 0.4f) + t * glm::vec3(1, 1, 1);

	image[gid] = c;
}

Renderer_cu::Renderer_cu(Renderer_cu&& o) {
	width = o.width;
	height = o.width;
	d_image = o.d_image;

	o.d_image = nullptr;
}
Renderer_cu& Renderer_cu::operator=(Renderer_cu&& o) {

	LOG(INFO) << "Renderer_cu::operator=(Renderer_cu&&) ==> Freeing device memory.";
	CUDA_ASSERT(cudaFree(d_image));
	d_image = nullptr;
	LOG(INFO) << "Renderer_cu::operator=(Renderer_cu&&) ==> memory freed.";

	width = o.width;
	height = o.height;
	d_image = o.d_image;

	o.d_image = nullptr;

	return *this;	
}

Renderer_cu::Renderer_cu(int width, int height) : width(width), height(height) {
	assert(width > 0 && height > 0);

	int kb_allocated = width * height * sizeof(glm::vec3) / 1000;
	LOG(INFO) << "Renderer_cu::Renderer_cu ==> Allocating " << kb_allocated << "KB for a " << width << "x" << height << " image on device.";
	CUDA_ASSERT(cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3)));
	LOG(INFO) << "Renderer_cu::Renderer_cu ==> allocation finished.";
}

Renderer_cu::~Renderer_cu() {

	LOG(INFO) << "Renderer_cu::~Renderer_cu ==> Deleting kernel device memory.";
	if (d_image == nullptr) LOG(INFO) << "Renderer_cu::~Renderer_cu ==> d_image has already been deleted.";

	CUDA_ASSERT(cudaFree(d_image));
	d_image = nullptr;

	LOG(INFO) << "Renderer_cu::~Renderer_cu ==> deletion finished.";
}

void Renderer_cu::Run() {

	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (width + threads.x - 1) / threads.x;
	blocks.y = (height + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer_cu::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y << ".";
	kernel<<<blocks, threads>>>(width, height, d_image);
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());
	LOG(INFO) << "Renderer_cu::Run ==> kernel finished.";
}

std::vector<glm::vec3> Renderer_cu::Download() {

	LOG(INFO) << "Renderer_cu::Download ==> Downloading kernel image from device.";
	std::vector<glm::vec3> h_image(width * height, glm::vec3(0.0f));
	CUDA_ASSERT(cudaMemcpy((glm::vec3*)h_image.data(), d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
	LOG(INFO) << "Renderer_cu::Download ==> download done.";

	return h_image;
}

} // namespace RT_ENGINE //