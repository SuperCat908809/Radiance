#include "triangle_bvh.h"

#include "cuError.h"
#include "host_timer.h"


using namespace RT_ENGINE;

TriangleBVH::TriangleBVH(TriangleBVH&& o) {
	d_nodes = o.d_nodes;
	d_tris = o.d_tris;
	d_indices = o.d_indices;

	root_idx = o.root_idx;
	nodes_used = o.nodes_used;

	o.d_nodes = nullptr;
	o.d_tris = nullptr;
	o.d_indices = nullptr;
}

TriangleBVH& TriangleBVH::operator=(TriangleBVH&& o) {
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));

	d_nodes = o.d_nodes;
	d_tris = o.d_tris;
	d_indices = o.d_indices;

	root_idx = o.root_idx;
	nodes_used = o.nodes_used;

	o.d_nodes = nullptr;
	o.d_tris = nullptr;
	o.d_indices = nullptr;

	return *this;
}

TriangleBVH::~TriangleBVH() {
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));
}


#define rnd (rand() / (float)RAND_MAX)
TriangleBVH::TriangleBVH(int tri_count, int seed) : tri_count(tri_count) {
	std::vector<Tri> triangles{};
	std::vector<int> indices{};

	srand(seed);

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Generating random triangles.";
	HostTimer tri_gen_timer{};
	tri_gen_timer.Start();

	for (int i = 0; i < tri_count; i++) {
		glm::vec3 r0(rnd, rnd, rnd);
		glm::vec3 r1(rnd, rnd, rnd);
		glm::vec3 r2(rnd, rnd, rnd);
		Tri tri{};
		tri.v0 = r0 * 9.0f - glm::vec3(5);
		tri.v1 = tri.v0 + r1;
		tri.v2 = tri.v0 + r2;
		tri.centeroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;

		triangles.push_back(tri);
		indices.push_back(triangles.size() - 1);
	}

	tri_gen_timer.End();
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> triangles generated in " << tri_gen_timer.ElapsedTimeMS() << "ms.";
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Starting BVH construction.";
	HostTimer bvh_construction_timer{};
	bvh_construction_timer.Start();

	std::vector<BVHNode> nodes{};

	BVHNode root{};
	root.leftFirst = 0;
	root.triCount = tri_count;
	nodes.push_back(root);
	nodes_used = 1;
	root_idx = nodes.size() - 1;

	_updateNodeBounds(nodes, triangles, indices, root_idx);
	_subdivide(nodes, triangles, indices, root_idx);

	bvh_construction_timer.End();
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> BVH built in " << bvh_construction_timer.ElapsedTimeMS() << "ms.";

	for (int i = 0; i < nodes.size(); i++) {
		BVHNode node = nodes[i];
		if (node.triCount != 0) {
			assert(node.leftFirst >= 0 && node.leftFirst <= tri_count);
			assert(node.leftFirst + node.triCount >= 0 && node.leftFirst + node.triCount <= tri_count);
		}
	}

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Copying BVH data to device.";

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << nodes_used * sizeof(BVHNode) / 1000 << "KB for BVH nodes.";
	CUDA_ASSERT(cudaMalloc((void**)&d_nodes, nodes_used * sizeof(BVHNode)));
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << triangles.size() * sizeof(Tri) / 1000 << "KB for triangles.";
	CUDA_ASSERT(cudaMalloc((void**)&d_tris, triangles.size() * sizeof(Tri)));
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << indices.size() * sizeof(int) / 1000 << "KB for triangle indices.";
	CUDA_ASSERT(cudaMalloc((void**)&d_indices, indices.size() * sizeof(int)));

	CUDA_ASSERT(cudaMemcpy(d_nodes, nodes.data(), nodes_used * sizeof(BVHNode), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice));

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> BVH data allocated and ready.";
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> #BVHNodes: " << nodes_used << ", #triangles and indices: " << tri_count << ".";
}

#undef min
#undef max

void TriangleBVH::_updateNodeBounds(std::vector<BVHNode>& nodes, std::vector<Tri>& triangles, std::vector<int> indices, int idx) {
	BVHNode& node = nodes[idx];
	node.aabbMin = glm::vec3(1e30f);
	node.aabbMax = glm::vec3(-1e30f);

	for (int i = 0; i < node.triCount; i++) {
		int triIdx = indices[node.leftFirst + i];
		const Tri& leafTri = triangles[triIdx];

		node.aabbMin = glm::min(node.aabbMin, leafTri.v0);
		node.aabbMin = glm::min(node.aabbMin, leafTri.v1);
		node.aabbMin = glm::min(node.aabbMin, leafTri.v2);

		node.aabbMax = glm::max(node.aabbMax, leafTri.v0);
		node.aabbMax = glm::max(node.aabbMax, leafTri.v1);
		node.aabbMax = glm::max(node.aabbMax, leafTri.v2);
	}
}

void TriangleBVH::_subdivide(std::vector<BVHNode>& nodes, std::vector<Tri>& triangles, std::vector<int> indices, int idx) {
#if 1
	BVHNode& node = nodes[idx];
	if (node.triCount <= 2) return;

	glm::vec3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;

	int i = node.leftFirst;
	int j = node.leftFirst + node.triCount - 1;

	while (i <= j) {
		if (triangles[indices[i]].centeroid[axis] < splitPos) {
			i++;
		}
		else {
			std::swap(indices[i], indices[j--]);
		}
	}

	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;

	// create child nodes
	int leftChildIdx = nodes.size();
	int rightChildIdx = nodes.size() + 1;
	BVHNode left_node{}, right_node{};
	left_node.leftFirst = node.leftFirst;
	left_node.triCount = leftCount;
	right_node.leftFirst = i;
	right_node.triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;

	nodes.push_back(left_node);
	nodes.push_back(right_node);
	nodes_used += 2;

	_updateNodeBounds(nodes, triangles, indices, leftChildIdx);
	_updateNodeBounds(nodes, triangles, indices, rightChildIdx);

	_subdivide(nodes, triangles, indices, leftChildIdx);
	_subdivide(nodes, triangles, indices, rightChildIdx);
#else
	// terminate recursion
	BVHNode& node = nodes[idx];
	if (node.triCount <= 2) return;
	// determine split axis and position
	glm::vec3 extent = node.aabbMax - node.aabbMin;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
	// in-place partition
	int i = node.leftFirst;
	int j = i + node.triCount - 1;
	while (i <= j)
	{
		if (triangles[indices[i]].centeroid[axis] < splitPos)
			i++;
		else
			std::swap(indices[i], indices[j--]);
	}
	// abort split if one of the sides is empty
	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;
	// create child nodes
	int leftChildIdx = nodes_used++;
	int rightChildIdx = nodes_used++;
	BVHNode left{}, right{};
	left.leftFirst = node.leftFirst;
	left.triCount = leftCount;
	right.leftFirst = i;
	right.triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;

	nodes.push_back(left);
	nodes.push_back(right);

	_updateNodeBounds(nodes, triangles, indices, leftChildIdx);
	_updateNodeBounds(nodes, triangles, indices, rightChildIdx);
	// recurse
	_subdivide(nodes, triangles, indices, leftChildIdx);
	_subdivide(nodes, triangles, indices, rightChildIdx);
#endif
}



TriangleBVH_2::~TriangleBVH_2() {
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));
}

TriangleBVH_2::TriangleBVH_2(TriangleBVH_2&& o) {
	d_tris = o.d_tris;
	d_nodes = o.d_nodes;
	d_indices = o.d_indices;

	root_node_idx = o.root_node_idx;
	triangle_count = o.triangle_count;
	nodes_used = o.nodes_used;

	nodes = std::move(o.nodes);
	triangles = std::move(o.triangles);
	indices = std::move(o.indices);

	o.d_tris = nullptr;
	o.d_nodes = nullptr;
	o.d_indices = nullptr;
}
TriangleBVH_2& TriangleBVH_2::operator=(TriangleBVH_2&& o) {
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));

	d_tris = o.d_tris;
	d_nodes = o.d_nodes;
	d_indices = o.d_indices;

	root_node_idx = o.root_node_idx;
	triangle_count = o.triangle_count;
	nodes_used = o.nodes_used;

	nodes = std::move(o.nodes);
	triangles = std::move(o.triangles);
	indices = std::move(o.indices);

	o.d_tris = nullptr;
	o.d_nodes = nullptr;
	o.d_indices = nullptr;

	return *this;
}

TriangleBVH_2::TriangleBVH_2(int triangle_count, int seed) : triangle_count(triangle_count) {
	triangles = {};
	indices = {};
	nodes = {};

	srand(seed);

	LOG(INFO) << "TriangleBVH_2::TriangleBVH_2 ==> Generating random triangles.";
	HostTimer tri_gen_timer{};
	tri_gen_timer.Start();

	for (int i = 0; i < triangle_count; i++) {
		glm::vec3 r0(rnd, rnd, rnd);
		glm::vec3 r1(rnd, rnd, rnd);
		glm::vec3 r2(rnd, rnd, rnd);
		Tri tri{};
		tri.v0 = r0 * 9.0f - glm::vec3(5);
		tri.v1 = tri.v0 + r1;
		tri.v2 = tri.v0 + r2;
		tri.centeroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;

		triangles.push_back(tri);
		indices.push_back(triangles.size() - 1);
	}

	tri_gen_timer.End();
	LOG(INFO) << "TriangleBVH_2::TriangleBVH_2 ==> triangles generated in " << tri_gen_timer.ElapsedTimeMS() << "ms.";
	LOG(INFO) << "TriangleBVH_2::TriangleBVH_2 ==> Starting BVH construction.";

	BVHNode root{};
	root.leftChild_firstTriangle = 0;
	root.triangle_count = triangle_count;
	nodes.push_back(root);
	root_node_idx = nodes.size() - 1;
	nodes_used = 1;
	
	_updateNodeBounds(root_node_idx);
	_subdivideNode(root_node_idx);

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> BVH built.";



	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Copying BVH data to device.";

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << nodes.size() * sizeof(BVHNode) / 1000 << "KB for BVH nodes.";
	CUDA_ASSERT(cudaMalloc((void**)&d_nodes, nodes.size() * sizeof(BVHNode)));
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << triangles.size() * sizeof(Tri) / 1000 << "KB for triangles.";
	CUDA_ASSERT(cudaMalloc((void**)&d_tris, triangles.size() * sizeof(Tri)));
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> Allocating " << indices.size() * sizeof(int) / 1000 << "KB for triangle indices.";
	CUDA_ASSERT(cudaMalloc((void**)&d_indices, indices.size() * sizeof(int)));

	CUDA_ASSERT(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice));

	LOG(INFO) << "TriangleBVH::TriangleBVH ==> BVH data allocated and ready.";
	LOG(INFO) << "TriangleBVH::TriangleBVH ==> #BVHNodes: " << nodes_used << ", #triangles and indices: " << triangle_count << ".";
}

void TriangleBVH_2::_updateNodeBounds(int node_index) {
	BVHNode& node = nodes[node_index];
	node.bounds = aabb{};

	for (int i = 0; i < node.triangle_count; i++) {
		int triangle_index = indices[node.leftChild_firstTriangle + i];
		const Tri& leaf_triangle = triangles[triangle_index];

		node.bounds.expand(leaf_triangle.v0);
		node.bounds.expand(leaf_triangle.v1);
		node.bounds.expand(leaf_triangle.v2);
	}
}

void TriangleBVH_2::_subdivideNode(int node_index) {
	BVHNode& node = nodes[node_index];
	if (node.triangle_count <= 2) return;

	glm::vec3 extent = node.bounds.extents();
	int axis = 0;
	if (extent.y > extent.x)axis = 1;
	if (extent.z > extent[axis])axis = 2;
	float splitPos = (node.bounds.max[axis] - node.bounds.min[axis]) / 2.0f;

	int i = node.leftChild_firstTriangle;
	int j = i + node.triangle_count - 1;

	while (i <= j) {
		if (triangles[indices[i]].centeroid[axis] < splitPos) {
			i++;
		}
		else {
			std::swap(indices[i], indices[j--]);
		}
	}

	int left_count = i = node.leftChild_firstTriangle;
	if (left_count == 0 || left_count == node.triangle_count) return;

	int leftChildIndex = nodes.size();
	int rightChildIndex = leftChildIndex + 1;

	BVHNode left_node{}, right_node{};
	left_node.leftChild_firstTriangle = node.leftChild_firstTriangle;
	left_node.triangle_count = left_count;
	right_node.leftChild_firstTriangle = i;
	right_node.triangle_count = node.triangle_count - left_node.triangle_count;

	node.leftChild_firstTriangle = leftChildIndex;
	node.triangle_count = 0;

	nodes.push_back(left_node);
	nodes.push_back(right_node);
	nodes_used += 2;

	_updateNodeBounds(leftChildIndex);
	_updateNodeBounds(rightChildIndex);

	_subdivideNode(leftChildIndex);
	_subdivideNode(rightChildIndex);
}