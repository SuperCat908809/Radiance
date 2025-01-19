#ifndef TRIANGLE_BVH_CLASS_CUDA_H
#define TRIANGLE_BVH_CLASS_CUDA_H

#include <vector>
#include <glm/glm.hpp>

#include "ray.h"
#include "triangles.h"


namespace RT_ENGINE {

#undef min
#undef max

class TriangleBVH {
	struct BVHNode {
		glm::vec3 aabbMin, aabbMax;
		int leftFirst, triCount;
		__device__ bool isLeaf() const { return triCount > 0; }
		__device__ bool intersect(const ray& r, const TraceRecord& rec) const {
#if 1
			glm::vec3 t1 = (aabbMin - r.o) / r.d, t2 = (aabbMax - r.o) / r.d;
			glm::vec3 tmin = glm::min(t1, t2), tmax = glm::max(t1, t2);

			float ttmin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
			float ttmax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

			return ttmax >= ttmin && ttmin < rec.t && ttmax > 0;
#else
			float tx1 = (aabbMin.x - r.o.x) / r.d.x, tx2 = (aabbMax.x - r.o.x) / r.d.x;
			float tmin = glm::min(tx1, tx2), tmax = glm::max(tx1, tx2);
			float ty1 = (aabbMin.y - r.o.y) / r.d.y, ty2 = (aabbMax.y - r.o.y) / r.d.y;
			tmin = glm::max(tmin, glm::min(ty1, ty2)), tmax = glm::min(tmax, glm::max(ty1, ty2));
			float tz1 = (aabbMin.z - r.o.z) / r.d.z, tz2 = (aabbMax.z - r.o.z) / r.d.z;
			tmin = glm::max(tmin, glm::min(tz1, tz2)), tmax = glm::min(tmax, glm::max(tz1, tz2));
			return tmax >= tmin && tmin < rec.t && tmax > 0;
#endif
		}
		__device__ float intersect2(const ray& r, const TraceRecord& rec) const {
			glm::vec3 t1 = (aabbMin - r.o) / r.d, t2 = (aabbMax - r.o) / r.d;
			glm::vec3 tmin = glm::min(t1, t2), tmax = glm::max(t1, t2);

			float ttmin = glm::max(glm::max(tmin.x, tmin.y), tmin.z);
			float ttmax = glm::min(glm::min(tmax.x, tmax.y), tmax.z);

			if (ttmax >= ttmin && ttmin < rec.t && ttmax > 0) {
				return ttmin;
			}
			return 1e10f;
		}
	};

	void _updateNodeBounds(std::vector<BVHNode>& nodes, std::vector<Tri>& tris, std::vector<int> indices, int node_idx);
	void _subdivide(std::vector<BVHNode>& nodes, std::vector<Tri>& tris, std::vector<int> indices, int node_idx);

	BVHNode* d_nodes;
	Tri* d_tris;
	int* d_indices;

	int root_idx, tri_count, nodes_used;

	TriangleBVH(const TriangleBVH&) = delete;
	TriangleBVH& operator=(const TriangleBVH&) = delete;

public:

	TriangleBVH(TriangleBVH&&);
	TriangleBVH& operator=(TriangleBVH&&);
	~TriangleBVH();

	struct handle_cu {
		BVHNode* d_nodes;
		Tri* d_tris;
		int* d_indices;
		int root_idx;
		int tri_count;
		int nodes_used;

		__device__ bool rec_intersect(const ray& r, TraceRecord& rec, int idx) const {
			if (idx < 0 || idx >= nodes_used) return false;
			const BVHNode* node = &d_nodes[idx];
			if (!node->intersect(r, rec)) return false;
			if (node->isLeaf()) {
				for (int i = 0; i < node->triCount; i++)
					intersect_tri(r, rec, d_tris[d_indices[node->leftFirst + i]]);
			}
			else
			{
				rec_intersect(r, rec, node->leftFirst);
				rec_intersect(r, rec, node->leftFirst + 1);
			}
		}
		__device__ bool intersect(const ray& r, TraceRecord& rec) const {
#if 0
			const BVHNode* node = &d_nodes[root_idx];
			if (!node->intersect(r, rec)) return false;

			bool hit_any = false;
			for (int i = 0; i < tri_count; i++) {
				hit_any |= intersect_tri(r, rec, d_tris[i]);
			}
			return hit_any;
#elif 0
			return rec_intersect(r, rec, root_idx);
#else
			bool hit_any = false;
			struct traversal_node { const BVHNode* node; int depth; };
			//const BVHNode* nodes[64];
			traversal_node nodes[64];
			int head = 0;
			nodes[head++] = { &d_nodes[root_idx], 0 };

			while (head > 0) {
				traversal_node node = nodes[--head];
				if (node.depth <= 1) if (!node.node->intersect(r, rec)) continue;
				if (node.node->isLeaf()) {
#if 1
					for (int i = 0; i < node.node->triCount; i++) {
						hit_any |= intersect_tri(r, rec, d_tris[d_indices[node.node->leftFirst + i]]);
					}
#else
					unsigned long long tmp = (unsigned long long)node;
					int col = 0;
						
					col ^= 0xff & (tmp >>  0);
					col ^= 0xff & (tmp >>  8);
					col ^= 0xff & (tmp >> 16);
					col ^= 0xff & (tmp >> 24);
					col ^= 0xff & (tmp >> 32);
					col ^= 0xff & (tmp >> 40);
					col ^= 0xff & (tmp >> 48);
					col ^= 0xff & (tmp >> 56);

					rec.t = node->intersect2(r, rec);
					rec.n = glm::vec3(0, col/(float)0xff, 0);
					return true;
#endif
				}
				else {
					nodes[head++] = { &d_nodes[node.node->leftFirst + 0], node.depth + 1 };
					nodes[head++] = { &d_nodes[node.node->leftFirst + 1], node.depth + 1 };
				}
			}

			return hit_any;
#endif
		}
	};

	TriangleBVH(int tri_count, int seed);

	handle_cu getDeviceHandle() const { return handle_cu{ d_nodes,d_tris,d_indices,root_idx,tri_count,nodes_used }; }

};

} // namespace RT_ENGINE

#endif // ifndef TRIANGLE_BVH_CLASS_CUDA_H //