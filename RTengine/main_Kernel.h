#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <inttypes.h>
#include <vector>


class Renderer_cu {

	int width, height;
	float* d_image{ nullptr };

public:
	Renderer_cu(int width, int height);

	void Run();
	std::vector<float> Download();
	void Delete();
};

#endif // MAIN_KERNEL_H //