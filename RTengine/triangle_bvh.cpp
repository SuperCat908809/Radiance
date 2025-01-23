#include "triangle_bvh.h"

#include <fstream>

#include "cuError.h"
#include "host_timer.h"


using namespace RT_ENGINE;

#undef min
#undef max
aabb::aabb() : min(1e30f), max(-1e30f) {}
aabb::aabb(glm::vec3 min, glm::vec3 max) : min(min), max(max) {}
void aabb::expand(const glm::vec3& p) { min = glm::min(min, p); max = glm::max(max, p); }
void aabb::expand(const aabb& b) { min = glm::min(min, b.min); max = glm::max(max, b.max); }
float aabb::surface_area() const { auto e = max - min; return 2.0f * (e.x * e.y + e.x * e.z + e.y * e.z); }

TriangleBVH::TriangleBVH(TriangleBVH&& o) noexcept {
	d_nodes = o.d_nodes;
	d_tris = o.d_tris;
	d_indices = o.d_indices;

	root_index = o.root_index;
	triangle_count = o.triangle_count;
	nodes_used = o.nodes_used;

	o.d_nodes = nullptr;
	o.d_tris = nullptr;
	o.d_indices = nullptr;
}

TriangleBVH& TriangleBVH::operator=(TriangleBVH&& o) noexcept {
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));

	d_nodes = o.d_nodes;
	d_tris = o.d_tris;
	d_indices = o.d_indices;

	root_index = o.root_index;
	triangle_count = o.triangle_count;
	nodes_used = o.nodes_used;

	o.d_nodes = nullptr;
	o.d_tris = nullptr;
	o.d_indices = nullptr;

	return *this;
}

TriangleBVH::~TriangleBVH() {
	if (d_nodes == nullptr && d_tris == nullptr && d_indices == nullptr) return;
	LOG(INFO) << "TriangleBVH::~TriangleBVH ==> Freeing device BVH memory.";
	if (d_nodes != nullptr) CUDA_ASSERT(cudaFree(d_nodes));
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	if (d_indices != nullptr) CUDA_ASSERT(cudaFree(d_indices));
	LOG(INFO) << "TriangleBVH::~TriangleBVH ==> deletion finished.";
}

TriangleBVH::TriangleBVH(BVHNode* dn, Tri* dt, int* di, int ri, int tc, int nu)
	: d_nodes(dn), d_tris(dt), d_indices(di), root_index(ri), triangle_count(tc), nodes_used(nu) {}
TriangleBVH TriangleBVH::Factory::BuildBVHFromRandomTriangles(int triangle_count, int seed) {

	TriangleBVH::Factory factory{};

	factory._generateTriangles(triangle_count, seed);
	factory._buildBVH();
	factory._loadToDevice();

	return TriangleBVH{ factory.d_nodes,factory.d_tris,factory.d_indices,factory.root_index,(int)factory.triangles.size(),(int)factory.bvh_nodes.size() };
}

TriangleBVH TriangleBVH::Factory::BuildBVHFromSimpleTri() {
	
	TriangleBVH::Factory factory{};

	factory._loadSimpleTri();
	factory._buildBVH();
	factory._loadToDevice();

	return TriangleBVH{ factory.d_nodes,factory.d_tris,factory.d_indices,factory.root_index,(int)factory.triangles.size(),(int)factory.bvh_nodes.size() };
}

void TriangleBVH::Factory::_loadToDevice() {
	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> Allocating and uploading BVH data to device.";

	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> Allocating " << bvh_nodes.size() * sizeof(BVHNode) / 1000 << "KB for BVH nodes.";
	CUDA_ASSERT(cudaMalloc((void**)&d_nodes, bvh_nodes.size() * sizeof(BVHNode)));
	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> Allocating " << triangles.size() * sizeof(Tri) / 1000 << "KB for triangles.";
	CUDA_ASSERT(cudaMalloc((void**)&d_tris, triangles.size() * sizeof(Tri)));
	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> Allocating " << triangle_indices.size() * sizeof(int) / 1000 << "KB for triangle indices.";
	CUDA_ASSERT(cudaMalloc((void**)&d_indices, triangle_indices.size() * sizeof(int)));

	CUDA_ASSERT(cudaMemcpy(d_nodes, bvh_nodes.data(), bvh_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_tris, triangles.data(), triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice));
	CUDA_ASSERT(cudaMemcpy(d_indices, triangle_indices.data(), triangle_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> BVH data allocated and ready.";
	LOG(INFO) << "TriangleBVH::Factory::_loadToDevice ==> #BVHNodes: " << bvh_nodes.size() << ", #triangles and indices: " << triangles.size() << ".";
}


#define rnd (rand() / (float)RAND_MAX)
void TriangleBVH::Factory::_generateTriangles(int triangle_count, int seed) {
	srand(seed);

	LOG(INFO) << "TriangleBVH::Factory::_generateTriangles ==> Generating random triangles.";
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
		triangle_indices.push_back(triangles.size() - 1);
	}

	tri_gen_timer.End();
	LOG(INFO) << "TriangleBVH::Factory::_generateTriangles ==> triangles generated in " << tri_gen_timer.ElapsedTimeMS() << "ms.";
}

void TriangleBVH::Factory::_loadSimpleTri() {
	LOG(INFO) << "TriangleBVH::Factory::_loadSimpleTri ==> loading simple triangle model from disk.";
	HostTimer model_load_timer{};
	model_load_timer.Start();

	FILE* file{nullptr};
	fopen_s(&file, "../Assets/test_geometry/unity.tri", "r");
	if (file == nullptr) {
		LOG(FATAL) << "Could not open model file.";
		assert(0);
		throw std::runtime_error("Could not open model file.");
	}

	float a, b, c, d, e, f, g, h, i;
	for (int t = 0; t < 12582; t++) {
		fscanf_s(file, "%f %f %f %f %f %f %f %f %f\n",
			&a, &b, &c, &d, &e, &f, &g, &h, &i);
		Tri tri{};
		tri.v0 = glm::vec3(a, b, c);
		tri.v1 = glm::vec3(d, e, f);
		tri.v2 = glm::vec3(g, h, i);
		tri.centeroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
		triangles.push_back(tri);
		triangle_indices.push_back(triangles.size() - 1);
	}
	fclose(file);

	model_load_timer.End();
	LOG(INFO) << "TriangleBVH::Factory::_loadSimpleTri ==> simple triangle model loaded " << triangles.size() << " triangles in " << model_load_timer.ElapsedTimeMS() << "ms.";
}

void TriangleBVH::Factory::_buildBVH() {
	LOG(INFO) << "TriangleBVH::Factory::_buildBVH ==> Starting BVH construction.";
	HostTimer bvh_construction_timer{};
	bvh_construction_timer.Start();

	BVHNode root{};
	root.leftFirst = 0;
	root.triCount = triangles.size();
	bvh_nodes.push_back(root);
	root_index = bvh_nodes.size() - 1;

	_updateNodeBounds(root_index);
	_subdivideNode(root_index);

	bvh_construction_timer.End();
	LOG(INFO) << "TriangleBVH::Factory::_buildBVH ==> BVH built in " << bvh_construction_timer.ElapsedTimeMS() << "ms.";
}

float TriangleBVH::Factory::_findBestSplitPlane(int node_index, int& axis, float& split_pos) {

	BVHNode& node = bvh_nodes[node_index];

	int best_axis = -1;
	float best_pos = 0.0f;
	float best_cost = 1e30f;

	for (int candidate_axis = 0; candidate_axis < 3; candidate_axis++) {
#if TARGET_BVH_ALGORITHM < BINNED_BVH_CONSTRUCTION_V1
		for (int i = 0; i < node.triCount; i++) {
			Tri& triangle = triangles[triangle_indices[node.leftFirst + i]];
			float candidate_pos = triangle.centeroid[candidate_axis];
			float candidate_cost = _evaluateSAH(node_index, candidate_axis, candidate_pos);

			if (candidate_cost < best_cost) {
				best_pos = candidate_pos;
				best_axis = candidate_axis;
				best_cost = candidate_cost;
			}
		}
#else
#if TARGET_BVH_ALGORITHM == BINNED_BVH_CONSTRUCTION_V1
		float bounds_min = node.bounds.min[candidate_axis];
		float bounds_max = node.bounds.max[candidate_axis];
#else
		float bounds_min = 1e30f, bounds_max = 1e-30f;
		for (int i = 0; i < node.triCount; i++) {
			Tri& triangle = triangles[triangle_indices[node.leftFirst + i]];
			bounds_min = glm::min(bounds_min, triangle.centeroid[candidate_axis]);
			bounds_max = glm::max(bounds_max, triangle.centeroid[candidate_axis]);
		}
#endif

		float scale = (bounds_max - bounds_min) / 100;
		if (scale == 0.0f) continue;


		for (int i = 1; i < 100; i++) {
			float candidate_pos = bounds_min + i * scale;
			float candidate_cost = _evaluateSAH(node_index, candidate_axis, candidate_pos);

			if (candidate_cost < best_cost) {
				best_pos = candidate_pos;
				best_axis = candidate_axis;
				best_cost = candidate_cost;
			}
		}
#endif
	}

	axis = best_axis;
	split_pos = best_pos;
	return best_cost;
}

float TriangleBVH::Factory::_evaluateSAH(int node_index, int candidate_axis, float candidate_split_pos) {

	BVHNode& node = bvh_nodes[node_index];

	aabb left_box{}, right_box{};
	int left_count = 0, right_count = 0;

	for (int i = 0; i < node.triCount; i++) {
		Tri& triangle = triangles[triangle_indices[node.leftFirst + i]];
		if (triangle.centeroid[candidate_axis] < candidate_split_pos) {
			left_count++;
			left_box.expand(triangle.v0);
			left_box.expand(triangle.v1);
			left_box.expand(triangle.v2);
		}
		else {
			right_count++;
			right_box.expand(triangle.v0);
			right_box.expand(triangle.v1);
			right_box.expand(triangle.v2);
		}
	}

	BVHNode left_tmp{ left_box,0, left_count };
	BVHNode right_tmp{ right_box,0, right_count };

	float cost = _calculateNodeCost(left_tmp) + _calculateNodeCost(right_tmp);
	return cost > 0 ? cost : 1e30f;
}

float TriangleBVH::Factory::_calculateNodeCost(const BVHNode& node) {
	return node.triCount * node.bounds.surface_area();
}

void TriangleBVH::Factory::_updateNodeBounds(int idx) {
	BVHNode& node = bvh_nodes[idx];
	node.bounds = {};

	for (int i = 0; i < node.triCount; i++) {
		int triIdx = triangle_indices[node.leftFirst + i];
		const Tri& leafTri = triangles[triIdx];

		node.bounds.expand(leafTri.v0);
		node.bounds.expand(leafTri.v1);
		node.bounds.expand(leafTri.v2);
	}
}

void TriangleBVH::Factory::_subdivideNode(int node_index) {
	BVHNode& node = bvh_nodes[node_index];
#if TARGET_BVH_ALGORITHM <= MIDDLE_SPLIT
	if (node.triCount <= 2) return;

	glm::vec3 extent = node.bounds.max - node.bounds.min;
	int axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	float split_pos = node.bounds.min[axis] + extent[axis] * 0.5f;
#else
	int axis{};
	float split_pos{};
	float nosplit_cost = _calculateNodeCost(node);
	float split_cost = _findBestSplitPlane(node_index, axis, split_pos);
	if (split_cost >= nosplit_cost) return;
#endif

	int i = node.leftFirst;
	int j = node.leftFirst + node.triCount - 1;

	while (i <= j) {
		if (triangles[triangle_indices[i]].centeroid[axis] < split_pos) {
			i++;
		}
		else {
			std::swap(triangle_indices[i], triangle_indices[j--]);
		}
	}

	int leftCount = i - node.leftFirst;
	if (leftCount == 0 || leftCount == node.triCount) return;

	// create child nodes
	int leftChildIdx = bvh_nodes.size();
	int rightChildIdx = bvh_nodes.size() + 1;
	BVHNode left_node{}, right_node{};
	left_node.leftFirst = node.leftFirst;
	left_node.triCount = leftCount;
	right_node.leftFirst = i;
	right_node.triCount = node.triCount - leftCount;
	node.leftFirst = leftChildIdx;
	node.triCount = 0;

	bvh_nodes.push_back(left_node);
	bvh_nodes.push_back(right_node);

	_updateNodeBounds(leftChildIdx);
	_updateNodeBounds(rightChildIdx);

	_subdivideNode(leftChildIdx);
	_subdivideNode(rightChildIdx);
}

TriangleBVH::handle_cu TriangleBVH::getDeviceHandle() const {
	return handle_cu{
		d_nodes,d_tris,d_indices,
		root_index,triangle_count,nodes_used
	};
}