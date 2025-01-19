#ifndef SCENE_CLASS_CUDA_H
#define SCENE_CLASS_CUDA_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <easylogging/easylogging++.h>

#include "cuError.h"


namespace RT_ENGINE {

struct Tri { glm::vec3 v0,v1,v2, centeroid;};

#define rnd (rand() / (float)RAND_MAX)

class Scene {

	Tri* d_tris;
	int tri_count;

	Scene(const Scene&) = delete;
	Scene operator=(const Scene&) = delete;

public:

	struct handle_cu {
		Tri* d_tris;
		int tri_count;
	};

	Scene(Scene&& o) {
		d_tris = o.d_tris;
		tri_count = o.tri_count;

		o.d_tris = nullptr;
	}
	Scene& operator=(Scene&& o) {
		if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));

		d_tris = o.d_tris;
		tri_count = o.tri_count;

		o.d_tris = nullptr;

		return *this;
	}

	~Scene() {
		if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
		d_tris = nullptr;
	}

	Scene(int tri_count, int seed) : tri_count(tri_count) {
		std::vector<Tri> triangles;

		srand(seed);

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
		}

		CUDA_ASSERT(cudaMalloc((void**)&d_tris, tri_count * sizeof(Tri)));
		CUDA_ASSERT(cudaMemcpy(d_tris, triangles.data(), tri_count * sizeof(Tri), cudaMemcpyHostToDevice));
	}

	handle_cu getDeviceHandle() const { return handle_cu{ d_tris, tri_count }; }
}; // class Scene //

#undef rnd

} // namespace RT_ENGINE //

#endif // ifndef SCENE_CLASS_CUDA_H //