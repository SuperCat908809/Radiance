#include "renderer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"


namespace RT_ENGINE {

__global__ void kernel(ColorRenderbuffer::handle_cu renderbuffer_handle, Scene::handle_cu scene_handle, Camera_cu cam) {

	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	if (gidx >= renderbuffer_handle.width || gidy >= renderbuffer_handle.height) return;

	float u = (gidx / (renderbuffer_handle.width - 1.0f)) * 2.0f - 1.0f;
	float v = (gidy / (renderbuffer_handle.height - 1.0f)) * 2.0f - 1.0f;

	ray r = cam.sample(u, v);
	TraceRecord rec{};

	bool hit = scene_handle.intersect(r, rec);

	glm::vec3 c{};
	if (hit) {
		float t = glm::dot(rec.n, glm::vec3(0, 1, 0));
		t = t * 0.5f + 0.5f;
		t = t * 0.8f + 0.1f;

		if (t > 1)
			c = glm::vec3(t, 0, 0);
		else if (t > 0)
			c = glm::vec3(t, t, t);
		else
			c = glm::vec3(0, 0, t);
	}
	else {
		float t = glm::normalize(r.d).y * 0.5f + 0.5f;
		c = (1 - t) * glm::vec3(0.1f, 0.2f, 0.4f) + t * glm::vec3(1, 1, 1);
	}

	renderbuffer_handle.at(gidx, gidy) = c;

} // kernel //

void _launch_kernel(
	dim3 blocks, dim3 threads,
	ColorRenderbuffer::handle_cu renderbuffer_handle, Scene::handle_cu scene_handle, Camera_cu cam) {

	kernel<<<blocks, threads>>>(renderbuffer_handle, scene_handle, cam);
}

} // namespace RT_ENGINE //