#ifndef RENDERER_CLASS_H
#define RENDERER_CLASS_H

#include <vector>
#include <glm/glm.hpp>

#include "renderbuffer.h"


namespace RT_ENGINE {
class Renderer {

	ColorRenderbuffer renderbuffer;

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

public:

	Renderer(Renderer&& o) : renderbuffer(std::move(o.renderbuffer)) {}
	Renderer& operator=(Renderer&& o) {
		renderbuffer = std::move(o.renderbuffer);
		return *this;
	}

	Renderer(int width, int height) : renderbuffer(width, height) {}
	~Renderer() = default;

	const ColorRenderbuffer& getRenderbuffer() const { return renderbuffer; }

	void Run();

}; // class Renderer //

} // namespace RT_ENGINE //

#ifdef RT_ENGINE_IMPLEMENTATION

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <easylogging/easylogging++.h>

#include "cuError.h"
#include "host_timer.h"
#include "cuda_timer.h"

#include "ray.h"
#include "camera.h"
#include "scene.h"

namespace RT_ENGINE {

__device__ bool hit_sphere(ray r) {

	glm::vec3 o(0, 0, 0);
	float radius = 1.0f;
	glm::vec3 oc = r.o - o;

	float a = glm::dot(r.d, r.d);
	float hb = glm::dot(r.d, oc);
	float c = glm::dot(oc, oc) - radius * radius;
	float d = hb * hb - a * c;

	return d > 0;

} // hit_sphere //

__global__ void kernel(ColorRenderbuffer::handle_cu renderbuffer_handle, Scene::handle_cu scene_handle, Camera_cu cam) {

	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (gidx >= renderbuffer_handle.width || gidy >= renderbuffer_handle.height) return;

	float u = (gidx / (renderbuffer_handle.width - 1.0f)) * 2.0f - 1.0f;
	float v = (gidy / (renderbuffer_handle.height - 1.0f)) * 2.0f - 1.0f;

	ray r = cam.sample(u, v);

	glm::vec3 c{};

	TraceRecord rec{};
	rec.t = 1e9f;

	for (int i = 0; i < scene_handle.tri_count; i++) {
		intersect_tri(r, rec, scene_handle.d_tris[i]);
	}

	if (rec.t < 1e9f) {
		//c = glm::vec3(1, 1, 1);
		float t = glm::dot(rec.n, glm::vec3(0, 1, 0)) * 0.8f + 0.1f;
		c = glm::vec3(t);
	}
	else {
		float t = glm::normalize(r.d).y * 0.5f + 0.5f;
		c = (1 - t) * glm::vec3(0.1f, 0.2f, 0.4f) + t * glm::vec3(1, 1, 1);
	}

	renderbuffer_handle.at(gidx, gidy) = c;

} // kernel //

void Renderer::Run() {

	glm::vec3 lookfrom(0, 0, -18);
	glm::vec3 lookat(0, 0, 0);
	glm::vec3 up(0, 1, 0);
	float vfov = glm::radians(36.0f);
	float aspect_ratio = renderbuffer.getWidth() / (float)renderbuffer.getHeight();

	Camera_cu cam(lookfrom, lookat, up, vfov, aspect_ratio);

	Scene scene(64, 0);

	CudaTimer render_kernel_timer{};
	HostTimer render_host_timer{};

	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (renderbuffer.getWidth() + threads.x - 1) / threads.x;
	blocks.y = (renderbuffer.getHeight() + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y << ".";
	render_kernel_timer.Start();
	render_host_timer.Start();
	kernel<<<blocks, threads>>>(renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam);
	render_host_timer.End();
	render_kernel_timer.End();
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());

	LOG(INFO) << "Renderer::Run ==> kernel finished in " << render_host_timer.ElapsedTimeMS() << "ms on host and " << render_kernel_timer.ElapsedTimeMS() << "ms on device.";
} // Renderer::Run //

} // namespace RT_ENGINE //
#endif // ifdef RT_ENGINE_IMPLEMENTATION //
#endif // ifndef RENDERER_CLASS_H //