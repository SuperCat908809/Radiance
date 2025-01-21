#include "renderer.h"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <easylogging/easylogging++.h>

#include "cuError.h"
#include "host_timer.h"
#include "cuda_timer.h"

#include "camera.h"
#include "scene.h"


namespace RT_ENGINE {
	void _launch_kernel(dim3, dim3, ColorRenderbuffer::handle_cu, Scene::handle_cu, Camera_cu);
}

using namespace RT_ENGINE;

Renderer::Renderer(Renderer&& o) noexcept : renderbuffer(std::move(o.renderbuffer)) {}
Renderer& Renderer::operator=(Renderer&& o) noexcept {
	renderbuffer = std::move(o.renderbuffer);
	return *this;
}
Renderer::~Renderer() = default;

Renderer::Renderer(int width, int height) : renderbuffer(width, height) {}

const ColorRenderbuffer& Renderer::getRenderbuffer() const { return renderbuffer; }

void Renderer::Run(float t) {

	glm::vec3 lookfrom(0, 0, -18);
	lookfrom = glm::rotate(lookfrom, t, glm::vec3(0, 1, 0));
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
	//kernel<<<blocks, threads>>>(renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam);
	_launch_kernel(blocks, threads, renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam);
	render_host_timer.End();
	render_kernel_timer.End();
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());

	LOG(INFO) << "Renderer::Run ==> kernel finished in " << render_host_timer.ElapsedTimeMS() << "ms on host and " << render_kernel_timer.ElapsedTimeMS() << "ms on device.";
} // Renderer::Run //