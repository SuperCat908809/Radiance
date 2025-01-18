#ifndef CUDA_UTILITIES_H
#define CUDA_UTILITIES_H

#include <string>
#include <cuda_runtime_api.h>


bool cuIsError(cudaError_t code);
std::string cuFormatErrorMessage(cudaError_t code);

#define GET_ERR_MSG(code) cuFormatErrorMessage(code)
#define CUDA_THROW(func) { auto cu_ret_error_code = func; if (cuIsError(cu_ret_error_code)) throw std::runtime_error(GET_ERR_MSG(cu_ret_error_code)); }
#ifdef NDEBUG
#define CUDA_ASSERT(func) (func)
#else
#define CUDA_ASSERT(func) { auto cu_ret_error_code = func; if (cuIsError(cu_ret_error_code)) { LOG(ERROR) << "CUDA_ASSERT ==> " << GET_ERR_MSG(cu_ret_error_code); assert(0); }}
#endif

#ifdef CUDA_ERROR_IMPLEMENTATION

#include <sstream>
#include <format>
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>


bool cuIsError(cudaError_t code) { return code != cudaSuccess; }

std::string cuFormatErrorMessage(cudaError_t code) {

	std::string error_name = cudaGetErrorName(code);
	std::string error_message = cudaGetErrorString(code);

	std::stringstream ss{};
	ss << "#" << code << ", " << error_name << " : " << error_message;
	return ss.str();
}

#endif // CUDA_ERROR_IMPLEMENTATION //

#endif // CUDA_UTILITIES_H //