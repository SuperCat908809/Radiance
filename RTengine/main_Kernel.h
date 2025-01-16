#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <inttypes.h>
#include <vector>

#include <glm/glm.hpp>


class Renderer_cu {

	int width, height;
	glm::vec3* d_image{ nullptr };

public:
	Renderer_cu(int width, int height);

	void Run();
	std::vector<glm::vec3> Download();
	void Delete();
};

#endif // MAIN_KERNEL_H //