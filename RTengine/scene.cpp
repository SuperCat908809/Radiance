#include "scene.h"

using namespace RT_ENGINE;

Scene::Scene(Scene&& o) noexcept : bvh(std::move(o.bvh)) {}
Scene& Scene::operator=(Scene&& o) noexcept {
	bvh = std::move(o.bvh);
	return *this;
}
Scene::~Scene() = default;

Scene::Scene() : bvh(std::move(TriangleBVH::Factory::BuildBVHFromSimpleTri())) {}
Scene::handle_cu Scene::getDeviceHandle() const { return handle_cu{ bvh.getDeviceHandle() }; }