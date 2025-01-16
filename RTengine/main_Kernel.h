#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <inttypes.h>
#include <vector>


class Kernel {

	int width, height;
	float* d_image{ nullptr };

public:
	Kernel(int width, int height);

	void Run();
	std::vector<float> Download();
	void Delete();
};

#endif // MAIN_KERNEL_H //