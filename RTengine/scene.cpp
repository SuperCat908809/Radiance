#include "scene.h"

using namespace RT_ENGINE;

Scene::Scene(Scene&& o) {
	d_tris = o.d_tris;
	tri_count = o.tri_count;

	o.d_tris = nullptr;
}
Scene& Scene::operator=(Scene&& o) {
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));

	d_tris = o.d_tris;
	tri_count = o.tri_count;

	o.d_tris = nullptr;

	return *this;
}

Scene::~Scene() {
	if (d_tris != nullptr) CUDA_ASSERT(cudaFree(d_tris));
	d_tris = nullptr;
}

#define rnd (rand() / (float)RAND_MAX)

Scene::Scene(int tri_count, int seed) : tri_count(tri_count) {
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