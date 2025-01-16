#ifndef MAIN_KERNEL_H
#define MAIN_KERNEL_H

#include <inttypes.h>

class Kernel {

public:
	Kernel() = default;

	int width = 256;
	int height = 256;

	void Run();
};

#endif // MAIN_KERNEL_H //