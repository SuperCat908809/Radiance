#ifndef CAMERA_CLASS_CUDA_H
#define CAMERA_CLASS_CUDA_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "ray.h"

namespace RT_ENGINE {

class Camera_cu {
	glm::vec3 origin, forward, horizontal, vertical;

public:

	__host__ __device__ Camera_cu(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 up, float vfov, float aspect_ratio) {
		float theta = vfov / 2.0f;
		float viewport_height = tanf(theta) * 2.0f;
		float viewport_width = viewport_height * aspect_ratio;

		origin = lookfrom;
		glm::vec3 w = glm::normalize(lookat - lookfrom);
		glm::vec3 u = glm::normalize(glm::cross(up, w));
		glm::vec3 v = glm::cross(w, u);

		forward = w;
		horizontal = u * viewport_width / 2.0f;
		vertical = v * viewport_height / 2.0f;

		// viewport dimensions are halfed here because the uv inputs are in the range [-1, 1].
	}

	__device__ ray sample(float u, float v) const {
		glm::vec3 d = forward + horizontal * u + vertical * v;
		return ray(origin, d);
	}

}; // class Camera_cu //

} // namespace RT_ENGINE //

#endif // ifndef CAMERA_CLASS_CUDA_H //