#include "scene.h"

using namespace RT_ENGINE;

Scene::Scene(Scene&& o) noexcept : bvh(std::move(o.bvh)) {}
Scene& Scene::operator=(Scene&& o) noexcept {
	bvh = std::move(o.bvh);
	return *this;
}
Scene::~Scene() = default;

Scene::Scene(int triangle_count, int seed) : bvh(std::move(TriangleBVH::Factory::BuildBVH(triangle_count, seed))) {}
Scene::handle_cu Scene::getDeviceHandle() const { return handle_cu{ bvh.getDeviceHandle() }; }