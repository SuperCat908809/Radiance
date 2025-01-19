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

	struct handle_cu {
		TriangleBVH::handle_cu bvh_handle;
		__device__ bool intersect(const ray& r, TraceRecord& rec) const { return bvh_handle.intersect(r, rec); }
	};

	Scene(Scene&& o);
	Scene& operator=(Scene&& o);
	~Scene();

	Scene(int tri_count, int seed);

	handle_cu getDeviceHandle() const { return handle_cu{ bvh.getDeviceHandle() }; }
}; // class Scene //

} // namespace RT_ENGINE //

#endif // ifndef SCENE_CLASS_CUDA_H //