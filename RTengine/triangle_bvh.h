#ifndef TRIANGLE_BVH_CLASS_CUDA_H
#define TRIANGLE_BVH_CLASS_CUDA_H

#include <vector>
#include <glm/glm.hpp>

#include "ray.h"
#include "triangles.h"


namespace RT_ENGINE {

#define MIDDLE_SPLIT 0
#define SAH_V1 1
#define SAH_V1_CLOSEST_CHILD 2
#define SAH_V1_CLOSEST_CHILD_V2 3
#define RAY_INV_D 4
#define FOUR_SPP 5
#define WARP_LOCALITY 6
#define WARP_LOCALITY_SPP 7
#define WARP_LOCALITY_SPP_SUBPIXEL_OFFSET 8

#define TARGET_BVH_ALGORITHM WARP_LOCALITY_SPP_SUBPIXEL_OFFSET

struct aabb{
	glm::vec3 min, max;

	aabb();
	aabb(glm::vec3 min, glm::vec3 max);

	void expand(const glm::vec3& p);
	void expand(const aabb& b);

	float surface_area() const;

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

	int root_index{};

	BVHNode* d_nodes{ nullptr };
	Tri* d_tris{ nullptr };
	int* d_indices{ nullptr };

	void _updateNodeBounds(int node_index);
	void _subdivideNode(int node_index);

	void _generateTriangles(int triangle_count, int seed);
	void _loadSimpleTri();

	float _findBestSplitPlane(int node_index, int& axis, float& split_pos);
	float _evaluateSAH(int node_index, int candidate_axis, float candidate_split_pos);
	float _calculateNodeCost(int node_index);

	void _buildBVH();
	void _loadToDevice();

	Factory() = default;

public:

	static TriangleBVH BuildBVHFromRandomTriangles(int triangle_count, int seed);
	static TriangleBVH BuildBVHFromSimpleTri();
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