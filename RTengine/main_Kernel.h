#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <vector>
#include <glm/glm.hpp>


namespace RT_ENGINE {
class Renderer {

	int width, height;
	glm::vec3* d_image;

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	
public:

	Renderer(Renderer&&);
	Renderer& operator=(Renderer&&);

	Renderer(int width, int height);
	~Renderer();

	void Run();
	std::vector<glm::vec3> Download();

}; // class Renderer //
} // namespace RT_ENGINE //

#endif // define MAIN_KERNEL_H //