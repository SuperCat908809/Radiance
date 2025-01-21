#ifndef RENDERER_CLASS_H
#define RENDERER_CLASS_H

#include <vector>
#include <glm/glm.hpp>

#include "renderbuffer.h"
#include "camera.h"
#include "scene.h"


namespace RT_ENGINE {
class Renderer {

	ColorRenderbuffer renderbuffer;
	Scene scene;
	Camera_cu cam;

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

public:

	Renderer(Renderer&& o) noexcept;
	Renderer& operator=(Renderer&& o) noexcept;
	~Renderer();

	Renderer(int width, int height);

	const ColorRenderbuffer& getRenderbuffer() const;

	void Run(float t);
	void RunFPSTest(int orbit_steps, int frames_per_step);

}; // class Renderer //
} // namespace RT_ENGINE //

#ifdef RT_ENGINE_IMPLEMENTATION
#include "renderer.inl"
#endif // ifdef RT_ENGINE_IMPLEMENTATION //

#endif // ifndef RENDERER_CLASS_H //