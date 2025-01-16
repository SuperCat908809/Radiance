#include <iostream>
#include <inttypes.h>

#include <stb/stb_image_write.h>


int main() {

	std::cout << "Main start.\n";

	uint8_t* image = new uint8_t[256 * 256 * 3];

	for (int y = 0; y < 256; y++) {
		for (int x = 0; x < 256; x++) {
			int idx = (y * 256 + x) * 3;

			image[idx + 0] = x;
			image[idx + 1] = y;
			image[idx + 2] = (x + y) / 2;
		}
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_jpg("testing.jpg", 256, 256, 3, image, 95);

	delete[] image;

	std::cout << "\n\nFinished.\n";

	return 0;
}