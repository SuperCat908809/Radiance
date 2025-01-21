#include "triangle_bvh.h"

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
	for (int i = 0; i < tri_count; i++) {
		hit_any |= intersect_tri(r, rec, d_tris[i]);
	}
#else
	bool hit_any = false;
	const BVHNode* nodes[64];
	int head = 0;
	nodes[head++] = &d_nodes[root_idx];

	while (head > 0) {
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

	return hit_any;
#endif
}

} // namespace RT_ENGINE //