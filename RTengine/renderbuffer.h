#ifndef RENDERBUFFER_CLASS_CUDA_H
#define RENDERBUFFER_CLASS_CUDA_H


#include <cuda_runtime.h>

#include <vector>
#include <glm/glm.hpp>

namespace RT_ENGINE {

class ColorRenderbuffer {
	glm::vec3* d_image;
	int width, height;

	ColorRenderbuffer(const ColorRenderbuffer&) = delete;
	ColorRenderbuffer& operator=(const ColorRenderbuffer&) = delete;

public:

	ColorRenderbuffer(ColorRenderbuffer&&);
	ColorRenderbuffer& operator=(ColorRenderbuffer&&);

	struct handle_cu {
		glm::vec3* d_image;
		const int width, height;

		__device__ glm::vec3& at(int x, int y);
		__device__ glm::vec3& at(int idx);
	};

	ColorRenderbuffer(int width, int height);
	~ColorRenderbuffer();

	int getWidth() const { return width; }
	int getHeight() const { return height; }
	handle_cu getDeviceHandle() const { return handle_cu{ d_image,width,height }; }

	void Download(std::vector<glm::vec3>& host_dst) const;

}; // class ColorRenderbuffer //

#ifdef RT_ENGINE_IMPLEMENTATION
__device__ glm::vec3& ColorRenderbuffer::handle_cu::at(int x, int y) { return d_image[y * width + x]; }
__device__ glm::vec3& ColorRenderbuffer::handle_cu::at(int idx) { return d_image[idx]; }
#endif // ifdef RT_ENGINE_IMPLEMENTATION //

} // namespace RT_ENGINE //
#endif // ifndef RENDERBUFFER_CLASS_CUDA_H //