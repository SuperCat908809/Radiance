#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <inttypes.h>
#include <vector>

#include <glm/glm.hpp>


class Renderer_cu {

	int width, height;
	glm::vec3* d_image;

	Renderer_cu(const Renderer_cu&) = delete;
	Renderer_cu& operator=(const Renderer_cu&) = delete;
	Renderer_cu(Renderer_cu&&);
	Renderer_cu& operator=(Renderer_cu&&);

public:
	Renderer_cu(int width, int height);
	~Renderer_cu();

	void Run();
	std::vector<glm::vec3> Download();
};

#endif // MAIN_KERNEL_H //