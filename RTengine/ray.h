#ifndef RT_ENGINE_RAY_STRUCT_H
#define RT_ENGINE_RAY_STRUCT_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>


namespace RT_ENGINE {

struct ray {
	struct {
		union { glm::vec3 o, origin; };
		union { glm::vec3 d, direction; };
		union { glm::vec3 inv_d, inverse_direction; };
	};

	__device__ ray() : o(), d() {}
	__device__ ray(glm::vec3 o, glm::vec3 d) : o(o), d(d), inv_d(1.0f / d) {}
	__device__ glm::vec3 at(float t) const { return o + d * t; }
}; // struct ray //

struct TraceRecord {
	float t{ 1e9f };
	float u{ 0.0f }, v{ 0.0f };
	glm::vec3 n{ 0,0,0 };
};

} // namespace RT_ENGINE //
#endif // define RT_ENGINE_RAY_STRUCT_H //