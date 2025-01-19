#include "scene.h"

using namespace RT_ENGINE;

Scene::Scene(Scene&& o) : bvh(std::move(o.bvh)) {}
Scene& Scene::operator=(Scene&& o) {
	bvh = std::move(o.bvh);
	return *this;
}

Scene::~Scene() = default;

#define rnd (rand() / (float)RAND_MAX)

Scene::Scene(int tri_count, int seed) : bvh(tri_count, seed) {}