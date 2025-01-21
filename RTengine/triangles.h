#ifndef TRIANGLES_STRUCT_CUDA_H
#define TRIANGLES_STRUCT_CUDA_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "ray.h"


namespace RT_ENGINE {

struct Tri { glm::vec3 v0, v1, v2, centeroid; };

__device__ inline bool intersect_tri(const ray& r, TraceRecord& rec, const Tri& tri) {
	const glm::vec3 edge1 = tri.v1 - tri.v0;
	const glm::vec3 edge2 = tri.v2 - tri.v0;
	const glm::vec3 h = glm::cross(r.d, edge2);
	const float a = glm::dot(edge1, h);
	if (a > -1e-4f && a < 1e-4f) return false; // ray parallel to triangle
	const float f = 1 / a;
	const glm::vec3 s = r.o - tri.v0;
	const float u = f * glm::dot(s, h);
	if (u < 0 || u > 1) return false;
	const glm::vec3 q = glm::cross(s, edge1);
	const float v = f * glm::dot(r.d, q);
	if (v < 0 || u + v > 1) return false;
	const float t = f * glm::dot(edge2, q);
	if (t > 1e-4f && t < rec.t) {
		rec.t = t;
		rec.u = u;
		rec.v = v;
		rec.n = glm::normalize(glm::cross(edge1, edge2));
		return true;
	}
	return false;
}

} // namespace RT_ENGINE //
#endif // ifndef TRIANGLES_STRUCT_CUDA_H //