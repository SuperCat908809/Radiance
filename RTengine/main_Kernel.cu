#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include <glm/glm.hpp>
#include <stb/stb_image_write.h>
#include <easylogging/easylogging++.h>

#include "cuError.h"

#define RT_ENGINE_IMPLEMENTATION
#include "main_Kernel.h"
#include "renderbuffer.h"
#include "ray.h"


namespace RT_ENGINE {

__global__ void kernel(ColorRenderbuffer::handle_cu renderbuffer_handle) {

	int width = renderbuffer_handle.width;
	int height = renderbuffer_handle.height;

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

	renderbuffer_handle.at(gid) = c;
}

Renderer::Renderer(Renderer&& o) : renderbuffer(std::move(o.renderbuffer)) {}
Renderer& Renderer::operator=(Renderer&& o) {

	renderbuffer = std::move(o.renderbuffer);

	return *this;	
}

Renderer::Renderer(int width, int height) : renderbuffer(width, height) {}

Renderer::~Renderer() = default;

void Renderer::Run() {

	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (renderbuffer.getWidth() + threads.x - 1) / threads.x;
	blocks.y = (renderbuffer.getHeight() + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y << ".";
	kernel<<<blocks, threads>>>(renderbuffer.getDeviceHandle());
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());
	LOG(INFO) << "Renderer::Run ==> kernel finished.";
}

} // namespace RT_ENGINE //