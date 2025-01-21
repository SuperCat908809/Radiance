#include "triangle_bvh.h"
#include <device_launch_parameters.h>

namespace RT_ENGINE {

__device__ bool aabb::intersect(const ray& r, const TraceRecord& rec) const {
	glm::vec3 t1 = (min - r.o) / r.d, t2 = (max - r.o) / r.d;
	glm::vec3 tmin = glm::min(t1, t2), tmax = glm::max(t1, t2);

	float ttmin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
	float ttmax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

	return ttmax >= ttmin && ttmin < rec.t && ttmax > 0;
}
__device__ float aabb::intersect_dist(const ray& r, const TraceRecord& rec) const {
	glm::vec3 t1 = (min - r.o) / r.d, t2 = (max - r.o) / r.d;
	glm::vec3 tmin = glm::min(t1, t2), tmax = glm::max(t1, t2);

	float ttmin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
	float ttmax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

	if (ttmax >= ttmin && ttmin < rec.t && ttmax > 0) {
		return ttmin;
	}
	else {
		return 1e30f;
	}
}

__device__ bool TriangleBVH::BVHNode::isLeaf() const { return triCount > 0; }

__device__ bool TriangleBVH::handle_cu::intersect(const ray& r, TraceRecord& rec) const {
#if 0
	bool hit_any = false;
	for (int i = 0; i < triangle_count; i++) {
		hit_any |= intersect_tri(r, rec, d_tris[i]);
	}
	return hit_any;
#else
	bool hit_any = false;
	const BVHNode* nodes[32];
	int head = 0;
	nodes[head++] = &d_nodes[root_index];

	int max_head = 0;

	while (head > 0) {

		if (max_head < head) max_head = head;

		const BVHNode* node = nodes[--head];
		if (!node->bounds.intersect(r, rec)) continue;
		if (node->isLeaf()) {
			for (int i = 0; i < node->triCount; i++) {
				hit_any |= intersect_tri(r, rec, d_tris[d_indices[node->leftFirst + i]]);
			}
		}
		else {
			nodes[head++] = &d_nodes[node->leftFirst + 0];
			nodes[head++] = &d_nodes[node->leftFirst + 1];
		}
	}

#if 0
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int gidx = blockDim.x * blockIdx.x + threadIdx.x;
		int gidy = blockDim.y * blockIdx.y + threadIdx.y;
		printf("%ix%i max_head: %i\n", gidx, gidy, max_head);
	}
#endif

	return hit_any;
#endif
}

} // namespace RT_ENGINE //