#ifndef SCENE_CLASS_CUDA_H
#define SCENE_CLASS_CUDA_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <easylogging/easylogging++.h>

#include "cuError.h"

#include "ray.h"
#include "triangles.h"
#include "triangle_bvh.h"


namespace RT_ENGINE {

class Scene {

	TriangleBVH bvh;

	Scene(const Scene&) = delete;
	Scene operator=(const Scene&) = delete;

public:

	struct handle_cu;

	Scene(Scene&& o) noexcept;
	Scene& operator=(Scene&& o) noexcept;
	~Scene();

	Scene();

	void Animate(float time);

	handle_cu getDeviceHandle() const;

}; // class Scene //

struct Scene::handle_cu {
	TriangleBVH::handle_cu bvh_handle;
	__device__ bool intersect(const ray& r, TraceRecord& rec) const;
};

#ifdef RT_ENGINE_IMPLEMENTATION
__device__ bool Scene::handle_cu::intersect(const ray& r, TraceRecord& rec) const { return bvh_handle.intersect(r, rec); }
#endif

} // namespace RT_ENGINE //

#endif // ifndef SCENE_CLASS_CUDA_H //