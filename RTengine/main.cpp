#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
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
		std::vector<glm::vec3> float_image(width * height);
		Renderer renderer(width, height);

		int count = 1;
		for (int i = 0; i < count; i++) {
			renderer.Run(glm::radians(i * 360.0f / (float)count));
			//renderer.Run(0.0f);
			renderer.getRenderbuffer().Download(float_image);

			std::vector<glm::u8vec3> image(float_image.size());
			std::transform(float_image.begin(), float_image.end(), image.begin(),
				[](glm::vec3 c) { return c * 255.0f; });

			std::stringstream ss{};
			//ss << "out/kernel_bvh_testing_" << std::setw(3) << std::setfill('0') << i + 1 << ".jpg";
			ss << "out/kernel_bvh_testing_002.jpg";
			std::string path = ss.str();

			stbi_flip_vertically_on_write(true);
			stbi_write_jpg(path.c_str(), width, height, 3, image.data(), 90);
		}
	}

	LOG(INFO) << "main ==> Resetting the cuda device.";
	cudaDeviceReset();

	LOG(INFO) << "main ==> Main application finished";

	return 0;
}