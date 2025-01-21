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

	int root_index, triangle_count, nodes_used;

	TriangleBVH(const TriangleBVH&) = delete;
	TriangleBVH& operator=(const TriangleBVH&) = delete;

	TriangleBVH(BVHNode*, Tri*, int*, int, int, int);

public:

	class Factory;
	struct handle_cu;

	TriangleBVH(TriangleBVH&&) noexcept;
	TriangleBVH& operator=(TriangleBVH&&) noexcept;
	~TriangleBVH();

	handle_cu getDeviceHandle() const;
};

struct TriangleBVH::BVHNode {
	aabb bounds{};
	int leftFirst{}, triCount{};
	__device__ bool isLeaf() const;
};

class TriangleBVH::Factory {
	std::vector<BVHNode> bvh_nodes;
	std::vector<Tri> triangles;
	std::vector<int> triangle_indices;

	int root_index;

	void _updateNodeBounds(int node_index);
	void _subdivideNode(int node_index);

	void _buildBVH(int triangle_count, int seed);

	Factory() = default;

public:

	static TriangleBVH BuildBVH(int triangle_count, int seed);
};

struct TriangleBVH::handle_cu {
	BVHNode* d_nodes;
	Tri* d_tris;
	int* d_indices;

	int root_index;
	int triangle_count;
	int nodes_used;

	__device__ bool intersect(const ray& r, TraceRecord& rec) const;
};

} // namespace RT_ENGINE


#ifdef RT_ENGINE_IMPLEMENTATION
#include "triangle_bvh.inl"
#endif

#endif // ifndef TRIANGLE_BVH_CLASS_CUDA_H //