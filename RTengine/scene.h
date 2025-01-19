#ifndef SCENE_CLASS_CUDA_H
#define SCENE_CLASS_CUDA_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <easylogging/easylogging++.h>

#include "cuError.h"

#include "ray.h"
#include "triangles.h"


namespace RT_ENGINE {

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

	Scene(Scene&& o);
	Scene& operator=(Scene&& o);
	~Scene();

	Scene(int tri_count, int seed);

	handle_cu getDeviceHandle() const { return handle_cu{ d_tris, tri_count }; }
}; // class Scene //

} // namespace RT_ENGINE //

#endif // ifndef SCENE_CLASS_CUDA_H //