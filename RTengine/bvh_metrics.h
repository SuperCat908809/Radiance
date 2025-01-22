#ifndef BVH_METRICS_CUDA_H
#define BVH_METRICS_CUDA_H

#include <cuda_runtime.h>

#include "easylogging/easylogging++.h"
#include "cuError.h"


namespace RT_ENGINE {

__managed__ int metrics_length;
__managed__ int metrics_width;
struct BVH_Metrics{
	int* box_tests;
	int* triangle_tests;
	int* max_depth;
	int* branches_encountered;
};

__managed__ BVH_Metrics g_bvh_metrics;

void create_bvh_metrics(int length, int width);
void log_bvh_metrics();
void reset_bvh_metrics();
void delete_bvh_metrics();

} // namespace RT_ENGINE //


#ifdef RT_ENGINE_IMPLEMENTATION

namespace RT_ENGINE {

void create_bvh_metrics(int length, int width) {
	metrics_length = length;
	metrics_width = width;

	g_bvh_metrics = {};

	CUDA_ASSERT(cudaMallocManaged((void**)&g_bvh_metrics.box_tests, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMallocManaged((void**)&g_bvh_metrics.triangle_tests, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMallocManaged((void**)&g_bvh_metrics.max_depth, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMallocManaged((void**)&g_bvh_metrics.branches_encountered, metrics_length * sizeof(int)));
}
void log_bvh_metrics() {

	int total_box_tests = 0;
	int total_triangle_tests = 0;
	int total_max_depth = 0;
	int total_branches_encountered = 0;

	for (int i = 0; i < metrics_length; i++) {
		total_box_tests += g_bvh_metrics.box_tests[i];
		total_triangle_tests += g_bvh_metrics.triangle_tests[i];
		total_max_depth += g_bvh_metrics.max_depth[i];
		total_branches_encountered += g_bvh_metrics.branches_encountered[i];
	}

	LOG(INFO)
		<< "," << total_box_tests / (float)metrics_length
		<< "," << total_triangle_tests / (float)metrics_length
		<< "," << total_max_depth / (float)metrics_length
		<< "," << total_branches_encountered / (float)metrics_length;

}
void reset_bvh_metrics() {
	CUDA_ASSERT(cudaMemset(g_bvh_metrics.box_tests, 0, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMemset(g_bvh_metrics.triangle_tests, 0, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMemset(g_bvh_metrics.max_depth, 0, metrics_length * sizeof(int)));
	CUDA_ASSERT(cudaMemset(g_bvh_metrics.branches_encountered, 0, metrics_length * sizeof(int)));
}
void delete_bvh_metrics() {
	CUDA_ASSERT(cudaFree(g_bvh_metrics.box_tests));
	CUDA_ASSERT(cudaFree(g_bvh_metrics.triangle_tests));
	CUDA_ASSERT(cudaFree(g_bvh_metrics.max_depth));
	CUDA_ASSERT(cudaFree(g_bvh_metrics.branches_encountered));

	g_bvh_metrics.box_tests = nullptr;
	g_bvh_metrics.triangle_tests = nullptr;
	g_bvh_metrics.max_depth = nullptr;
	g_bvh_metrics.branches_encountered = nullptr;
}

} // namespace RT_ENGINE //

#endif // ifdef RT_ENGINE_IMPLEMENTATION //

#endif // endif BVH_METRICS_CUDA_H //