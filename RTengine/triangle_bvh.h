#ifndef TRIANGLE_BVH_CLASS_CUDA_H
#define TRIANGLE_BVH_CLASS_CUDA_H

#include <vector>
#include <glm/glm.hpp>

#include "ray.h"
#include "triangles.h"


namespace RT_ENGINE {

struct aabb{
	glm::vec3 min, max;

	aabb();
	aabb(glm::vec3 min, glm::vec3 max);

	void expand(const glm::vec3& p);
	void expand(const aabb& b);

	__device__ bool intersect(const ray& r, const TraceRecord& rec) const;
	__device__ float intersect_dist(const ray& r, const TraceRecord& rec) const;
};

class TriangleBVH {
	struct BVHNode;
	BVHNode* d_nodes;
	Tri* d_tris;
	int* d_indices;

	int root_idx, tri_count, nodes_used;

	void _updateNodeBounds(std::vector<BVHNode>& nodes, std::vector<Tri>& tris, std::vector<int>& indices, int node_idx);
	void _subdivide(std::vector<BVHNode>& nodes, std::vector<Tri>& tris, std::vector<int>& indices, int node_idx);

	TriangleBVH(const TriangleBVH&) = delete;
	TriangleBVH& operator=(const TriangleBVH&) = delete;

public:

#if 0
	class Factory {
		std::vector<BVHNode> bvh_nodes;
		std::vector<Tri> triangles;
		std::vector<int> triangle_indices;

		int root_index;

	public:

		static TriangleBVH BuildBVH(int triangle_count, int seed);
	};
#endif

	TriangleBVH(TriangleBVH&&);
	TriangleBVH& operator=(TriangleBVH&&);
	~TriangleBVH();

	TriangleBVH(int tri_count, int seed);

	struct handle_cu;
	handle_cu getDeviceHandle() const;
};

struct TriangleBVH::BVHNode {
	aabb bounds{};
	int leftFirst{}, triCount{};
	__device__ bool isLeaf() const;
};

struct TriangleBVH::handle_cu {
	BVHNode* d_nodes;
	Tri* d_tris;
	int* d_indices;
	int root_idx;
	int tri_count;
	int nodes_used;

	__device__ bool intersect(const ray& r, TraceRecord& rec) const;
};

} // namespace RT_ENGINE


#ifdef RT_ENGINE_IMPLEMENTATION
#include "triangle_bvh.inl"
#endif

#endif // ifndef TRIANGLE_BVH_CLASS_CUDA_H //