#include <iostream>
#include <cuda_runtime_api.h>

#include <stb/stb_image_write.h>

#include "main_Kernel.h"


int main() {

	std::cout << "Main start.\n";

	int width = 1920;
	int height = 1080;

	Renderer_cu kernel(width, height);

	kernel.Run();
	auto float_image = kernel.Download();
	kernel.Delete();

	std::vector<uint8_t> image(float_image.size());

	for (int i = 0; i < width * height * 3; i++) {
		image[i] = static_cast<uint8_t>(float_image[i] * 255.0f);
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_jpg("kernel_float_testing.jpg", width, height, 3, image.data(), 90);


	cudaDeviceReset();

	std::cout << "\n\nFinished.\n";

	return 0;
}