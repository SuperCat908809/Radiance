#include "triangle_bvh.h"
#include <device_launch_parameters.h>

#include "bvh_metrics.h"


namespace RT_ENGINE {

template <typename T> __device__ inline void cuda_swap(T& a, T& b) {
	T tmp = std::move(a);
	a = std::move(b);
	b = std::move(tmp);
}

#undef min
#undef max

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

	BVH_METRIC_INIT;

	bool hit_any = false;

	int head = 0;
	const BVHNode* nodes[32];
	nodes[head++] = &d_nodes[root_index];

	BVH_METRIC_STATEMENT(
	int depths[32];
	depths[0] = 0;
	);

	if (!nodes[0]->bounds.intersect(r, rec)) return false;
	BVH_METRIC_ADD_BOX_TESTS(1);
	// put afterwards s.t. those that miss the BVH will not contribute to the statistical analysis

	while (head > 0) {

		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);

		const BVHNode* node = nodes[--head];

		BVH_METRIC_STATEMENT(int current_depth = depths[head]);
		BVH_METRIC_MAX_DEPTH(current_depth);

#if TARGET_BVH_ALGORITHM >= SAH_V1_CLOSEST_CHILD_V2
		BVH_METRIC_ADD_BOX_TESTS(1);
		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);
		if (!node->bounds.intersect(r, rec)) continue;
#endif
		
		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);
		if (node->isLeaf()) {

			BVH_METRIC_ADD_TRIANGLE_TEST(node->triCount);
			BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(node->triCount);

			for (int i = 0; i < node->triCount; i++) {
				hit_any |= intersect_tri(r, rec, d_tris[d_indices[node->leftFirst + i]]);
			}
			continue;
		}

		BVHNode* left_node = &d_nodes[node->leftFirst + 0];
		BVHNode* right_node = &d_nodes[node->leftFirst + 1];

#if TARGET_BVH_ALGORITHM < SAH_V1_CLOSEST_CHILD
		BVH_METRIC_STATEMENT(depths[head] = current_depth + 1);
		nodes[head++] = left_node;
		BVH_METRIC_STATEMENT(depths[head] = current_depth + 1);
		nodes[head++] = right_node;
#else

		BVH_METRIC_ADD_BOX_TESTS(2);
		float left_dist = left_node->bounds.intersect_dist(r, rec);
		float right_dist = right_node->bounds.intersect_dist(r, rec);

		if (left_dist > right_dist) { cuda_swap(left_dist, right_dist); cuda_swap(left_node, right_node); }
		if (right_dist < 1e30f) {
			BVH_METRIC_STATEMENT(depths[head] = current_depth + 1);
			nodes[head++] = right_node;
		}
		if (left_dist < 1e30f) {
			BVH_METRIC_STATEMENT(depths[head] = current_depth + 1);
			nodes[head++] = left_node;
		}

		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(3);

#endif
	}

	return hit_any;
}

} // namespace RT_ENGINE //