#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime_api.h>

#include <stb/stb_image_write.h>
#include <easylogging/easylogging++.h>

#include "renderer.h"


using namespace RT_ENGINE;

int main() {

	LOG(INFO) << "main ==> Main application start";

	int width = 1920;
	int height = 1080;

	{
		Renderer renderer(width, height);

		renderer.Run();
		std::vector<glm::vec3> float_image(width * height);
		renderer.getRenderbuffer().Download(float_image);

		std::vector<glm::u8vec3> image(float_image.size());
		std::transform(float_image.begin(), float_image.end(), image.begin(),
			[](glm::vec3 c) { return c * 255.0f; });

		stbi_flip_vertically_on_write(true);
		stbi_write_jpg("kernel_raii_testing.jpg", width, height, 3, image.data(), 90);
	}

	LOG(INFO) << "main ==> Resetting the cuda device.";
	cudaDeviceReset();

	LOG(INFO) << "main ==> Main application finished";

	return 0;
}