#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <vector>
#include <glm/glm.hpp>


namespace RT_ENGINE {
class Renderer_cu {

	int width, height;
	glm::vec3* d_image;

	Renderer_cu(const Renderer_cu&) = delete;
	Renderer_cu& operator=(const Renderer_cu&) = delete;
	
public:

	Renderer_cu(Renderer_cu&&);
	Renderer_cu& operator=(Renderer_cu&&);

	Renderer_cu(int width, int height);
	~Renderer_cu();

	void Run();
	std::vector<glm::vec3> Download();

}; // class Renderer_cu //
} // namespace RT_ENGINE //

#endif // define MAIN_KERNEL_H //