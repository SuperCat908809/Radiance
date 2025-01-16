#include <iostream>

#include "main_Kernel.h"


int main() {

	std::cout << "Main start.\n";

	Kernel kernel{};

	kernel.width = 1920;
	kernel.height = 1080;

	kernel.Run();

	std::cout << "\n\nFinished.\n";

	return 0;
}