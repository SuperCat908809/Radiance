#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <vector>
#include <glm/glm.hpp>

#include "renderbuffer.h"


namespace RT_ENGINE {
class Renderer {

	ColorRenderbuffer renderbuffer;

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;
	
public:

	Renderer(Renderer&&);
	Renderer& operator=(Renderer&&);

	Renderer(int width, int height);
	~Renderer();

	const ColorRenderbuffer& getRenderbuffer() const { return renderbuffer; }

	void Run();

}; // class Renderer //
} // namespace RT_ENGINE //

#endif // ifndef MAIN_KERNEL_H //