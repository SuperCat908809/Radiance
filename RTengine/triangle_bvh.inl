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
#if 0
	bool hit_any = false;
	for (int i = 0; i < triangle_count; i++) {
		hit_any |= intersect_tri(r, rec, d_tris[i]);
	}
	return hit_any;
#else

#if 1
	BVH_METRIC_INIT;
#else
	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	int gid = (gidy / 8) * metrics_width + (gidx / 8);
	bool mt = ((threadIdx.x == 0) && (threadIdx.y == 0)); // mt == metric target
#endif

	bool hit_any = false;
	const BVHNode* nodes[32];
	int depths[32];

	int head = 0;
	depths[head] = 0;
	nodes[head++] = &d_nodes[root_index];

	if (!nodes[0]->bounds.intersect(r, rec)) return false;
	BVH_METRIC_ADD_BOX_TESTS(1);

	int max_head = 0;

	while (head > 0) {

		//if (mt) g_bvh_metrics.branches_encountered[gid]++;
		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);


		if (max_head < head) max_head = head;

		const BVHNode* node = nodes[--head];
		int current_depth = depths[head];

		BVH_METRIC_MAX_DEPTH(current_depth);
		//if (mt) if (current_depth > g_bvh_metrics.max_depth[gid]) g_bvh_metrics.max_depth[gid] = current_depth;

		BVH_METRIC_ADD_BOX_TESTS(1);
		//if (mt) g_bvh_metrics.box_tests[gid]++;
		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);
		//if (mt) g_bvh_metrics.branches_encountered[gid]++;

#if TARGET_BVH_ALGORITHM < SAH_V1_CLOSEST_CHILD_V2
		if (!node->bounds.intersect(r, rec)) continue;
#endif
		
		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(1);
		//if (mt) g_bvh_metrics.branches_encountered[gid]++;

		if (node->isLeaf()) {
			BVH_METRIC_ADD_TRIANGLE_TEST(node->triCount);
			//if (mt) g_bvh_metrics.triangle_tests[gid] += node->triCount;
			BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(node->triCount);
			//if (mt) g_bvh_metrics.branches_encountered[gid] += node->triCount;
			for (int i = 0; i < node->triCount; i++) {
				hit_any |= intersect_tri(r, rec, d_tris[d_indices[node->leftFirst + i]]);
			}
			continue;
		}

		BVHNode* left_node = &d_nodes[node->leftFirst + 0];
		BVHNode* right_node = &d_nodes[node->leftFirst + 1];

#if TARGET_BVH_ALGORITHM < SAH_V1_CLOSEST_CHILD
		depths[head] = current_depth + 1;
		nodes[head++] = left_node;
		depths[head] = current_depth + 1;
		nodes[head++] = right_node;
#else
		float left_dist = left_node->bounds.intersect_dist(r, rec);
		float right_dist = right_node->bounds.intersect_dist(r, rec);
		BVH_METRIC_ADD_BOX_TESTS(2);
		//if (mt) g_bvh_metrics.box_tests[gid] += 2;


		if (left_dist > right_dist) { cuda_swap(left_dist, right_dist); cuda_swap(left_node, right_node); }
		if (right_dist < 1e30f) {
			depths[head] = current_depth + 1;
			nodes[head++] = right_node;
		}
		if (left_dist < 1e30f) {
			depths[head] = current_depth + 1;
			nodes[head++] = left_node;
		}

		BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(3);
		//if (mt) g_bvh_metrics.branches_encountered[gid] += 3;

#endif
	}

#if 0
	if (threadIdx.x == 0 && threadIdx.y == 0 && max_head > 1)
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