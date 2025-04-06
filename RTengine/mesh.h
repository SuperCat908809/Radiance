#ifndef MESH_CLASS_H
#define MESH_CLASS_H

#include <vector>
#include <glm/glm.hpp>

#include "triangles.h"


namespace RT_ENGINE {
class Mesh {
	glm::vec3* d_vertices{};
	int* d_indices{};
};
} // namespace RT_ENGINE //

#endif // ifndef MESH_CLASS_H //