#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include <glm/glm.hpp>
#include <stb/stb_image_write.h>
#include <easylogging/easylogging++.h>

#include "cuError.h"

#define RT_ENGINE_IMPLEMENTATION
#include "main_Kernel.h"
#include "renderer.h"


namespace RT_ENGINE {

} // namespace RT_ENGINE //