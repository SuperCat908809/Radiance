#include "renderer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <curand.h>
#include <curand_kernel.h>

#include "ray.h"
#include "camera.h"
#include "scene.h"


namespace RT_ENGINE {

__global__ void kernel(ColorRenderbuffer::handle_cu renderbuffer_handle, Scene::handle_cu scene_handle, Camera_cu cam, int samples) {

#if TARGET_BVH_ALGORITHM < WARP_LOCALITY
	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
#else
	int gidx = blockIdx.x;
	int gidy = blockIdx.y;
#endif
	if (gidx >= renderbuffer_handle.width || gidy >= renderbuffer_handle.height) return;

	float u = (gidx / (renderbuffer_handle.width - 1.0f)) * 2.0f - 1.0f;
	float v = (gidy / (renderbuffer_handle.height - 1.0f)) * 2.0f - 1.0f;

	curandState_t random_state{};
#if TARGET_BVH_ALGORITHM < WARP_LOCALITY
	curand_init(gidy * renderbuffer_handle.width + gidx, 0, 0, &random_state);
#else
	int seed = gidy * renderbuffer_handle.width + gidx;
	seed = seed * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	curand_init(seed, 0, 0, &random_state);
#endif

	for (int s = 0; s < samples; s++) {

		glm::vec2 offset = glm::vec2(curand_uniform(&random_state), curand_uniform(&random_state)) * 2.0f - 1.0f;
		offset *= glm::vec2(1.0f / (renderbuffer_handle.width - 1.0f), 1.0f / (renderbuffer_handle.height - 1.0f)) * 0.5f * 0.8f;

#if TARGET_BVH_ALGORITHM < WARP_LOCALITY_SPP_SUBPIXEL_OFFSET
		ray r = cam.sample(u, v);
#else
		ray r = cam.sample(u + offset.x, v + offset.y);
#endif
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

#if TARGET_BVH_ALGORITHM < WARP_LOCALITY
		renderbuffer_handle.at(gidx, gidy) = c;
#else
		// only thread (0,0) in the block can write to the output
		if (threadIdx.x == 0 && threadIdx.y == 0)
			renderbuffer_handle.at(gidx, gidy) = c;
#endif
	}
} // kernel //

void _launch_kernel(
	dim3 blocks, dim3 threads,
	ColorRenderbuffer::handle_cu renderbuffer_handle, Scene::handle_cu scene_handle, Camera_cu cam, int samples) {

	kernel<<<blocks, threads>>>(renderbuffer_handle, scene_handle, cam, samples);
}

} // namespace RT_ENGINE //