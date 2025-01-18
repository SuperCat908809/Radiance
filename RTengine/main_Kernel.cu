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
#include "camera.h"

#include "cuda_timer.h"


namespace RT_ENGINE {

__device__ bool hit_sphere(ray r){
	glm::vec3 o(0, 0, 0);
	float radius = 1.0f;
	glm::vec3 oc = r.o - o;

	float a = glm::dot(r.d, r.d);
	float hb = glm::dot(r.d, oc);
	float c = glm::dot(oc, oc) - radius * radius;
	float d = hb * hb - a * c;

	return d > 0;
}

__global__ void kernel(ColorRenderbuffer::handle_cu renderbuffer_handle, Camera_cu cam) {

	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (gidx >= renderbuffer_handle.width || gidy >= renderbuffer_handle.height) return;

	float u = (gidx / (renderbuffer_handle.width - 1.0f)) * 2.0f - 1.0f;
	float v = (gidy / (renderbuffer_handle.height - 1.0f)) * 2.0f - 1.0f;

	ray r = cam.sample(u, v);

	glm::vec3 c{};

	if (hit_sphere(r)) {
		c = glm::vec3(1, 0, 0);	
	}
	else {
		float t = glm::normalize(r.d).y * 0.5f + 0.5f;
		c = (1 - t) * glm::vec3(0.1f, 0.2f, 0.4f) + t * glm::vec3(1, 1, 1);
	}
	renderbuffer_handle.at(gidx, gidy) = c;
}

Renderer::Renderer(Renderer&& o) : renderbuffer(std::move(o.renderbuffer)) {}
Renderer& Renderer::operator=(Renderer&& o) {

	renderbuffer = std::move(o.renderbuffer);

	return *this;	
}

Renderer::Renderer(int width, int height) : renderbuffer(width, height) {}

Renderer::~Renderer() = default;

void Renderer::Run() {

	glm::vec3 lookfrom(0, 0, -4);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float vfov = glm::radians(90.0f);
	float aspect_ratio = renderbuffer.getWidth() / (float)renderbuffer.getHeight();

	Camera_cu cam(lookfrom, lookat, up, vfov, aspect_ratio);

	CudaTimer render_kernel_timer{};

	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (renderbuffer.getWidth() + threads.x - 1) / threads.x;
	blocks.y = (renderbuffer.getHeight() + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y << ".";
	render_kernel_timer.Start();
	kernel<<<blocks, threads>>>(renderbuffer.getDeviceHandle(), cam);
	render_kernel_timer.End();
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());

	LOG(INFO) << "Renderer::Run ==> kernel finished in " << render_kernel_timer.ElapsedTime() << "ms.";
}

} // namespace RT_ENGINE //