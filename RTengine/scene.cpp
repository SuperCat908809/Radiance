#include "scene.h"

using namespace RT_ENGINE;

Scene::Scene(Scene&& o) noexcept : bvh(std::move(o.bvh)) {}
Scene& Scene::operator=(Scene&& o) noexcept {
	bvh = std::move(o.bvh);
	return *this;
}
Scene::~Scene() = default;

Scene::Scene() : bvh(std::move(TriangleBVH::Factory::BuildBVHFromBigBenTri(0.0f))) {}

void Scene::Animate(float time) {
	bvh = std::move(TriangleBVH::Factory::BuildBVHFromBigBenTri(time));
}
Scene::handle_cu Scene::getDeviceHandle() const { return handle_cu{ bvh.getDeviceHandle() }; }