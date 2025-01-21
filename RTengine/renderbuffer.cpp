#include "renderbuffer.h"
#include <easylogging/easylogging++.h>

#include <cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "cuError.h"


using namespace RT_ENGINE;

ColorRenderbuffer::ColorRenderbuffer(ColorRenderbuffer&& o) {
	width = o.width;
	height = o.height;
	d_image = o.d_image;

	o.d_image = nullptr;
}

ColorRenderbuffer& ColorRenderbuffer::operator=(ColorRenderbuffer&& o) {
	LOG(INFO) << "ColorRenderbuffer::operator=(ColorRenderbuffer&&) ==> Freeing destination device renderbuffer memory.";
	CUDA_ASSERT(cudaFree(d_image));
	LOG(INFO) << "ColorRenderbuffer::operator=(ColorRenderbuffer&&) ==> buffer freed.";

	width = o.width;
	height = o.height;
	d_image = o.d_image;

	o.d_image = nullptr;

	return *this;
}


ColorRenderbuffer::ColorRenderbuffer(int width, int height) : width(width), height(height) {
	assert(width > 0 && height > 0);

	int kb_allocated = width * height * sizeof(glm::vec3) / 1000;
	LOG(INFO) << "ColorRenderbuffer::ColorRenderbuffer ==> Allocating " << kb_allocated << "KB for a " << width << "x" << height << " renderbuffer on device.";
	CUDA_ASSERT(cudaMalloc((void**)&d_image, width * height * sizeof(glm::vec3)));
	LOG(INFO) << "ColorRenderbuffer::ColorRenderbuffer ==> allocation finished.";
}

ColorRenderbuffer::~ColorRenderbuffer() {
	LOG(INFO) << "ColorRenderbuffer::~ColorRenderbuffer ==> Freeing device renderbuffer memory.";
	if (d_image == nullptr) LOG(INFO) << "ColorRenderbuffer::~ColorRenderbuffer ==> d_image has already been freed.";

	CUDA_ASSERT(cudaFree(d_image));
	d_image = nullptr;

	LOG(INFO) << "ColorRenderbuffer::~ColorRenderbuffer ==> deletion finished.";
}

void ColorRenderbuffer::Download(std::vector<glm::vec3>& host_dst) const {

	assert(host_dst.size() == width * height);

	//LOG(INFO) << "ColorRenderbuffer::Download ==> Downloading renderbuffer to from device to host.";
	CUDA_ASSERT(cudaMemcpy((glm::vec3*)host_dst.data(), d_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
	//LOG(INFO) << "ColorRenderbuffer::Download ==> download done.";

}