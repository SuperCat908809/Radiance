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

struct thread_bvh_metric_reference {
	int& box_tests;
	int& triangle_tests;
	int& max_depth;
	int& branches_encountered;
};

__device__ thread_bvh_metric_reference getThreadBVHMetricReference();

#define BVH_METRIC_INIT thread_bvh_metric_reference metric_ref = getThreadBVHMetricReference()
#define BVH_METRIC_ADD_BOX_TESTS(n) metric_ref.box_tests += n
#define BVH_METRIC_ADD_TRIANGLE_TEST(n) metric_ref.triangle_tests += n;
#define BVH_METRIC_MAX_DEPTH(candidate_depth) if (metric_ref.max_depth < candidate_depth) metric_ref.max_depth = candidate_depth
#define BVH_METRIC_ADD_BRANCHES_ENCOUNTERED(n) metric_ref.branches_encountered += n

__managed__ BVH_Metrics g_bvh_metrics;

void create_bvh_metrics(int length, int width);
void log_bvh_metrics();
void reset_bvh_metrics();
void delete_bvh_metrics();

} // namespace RT_ENGINE //


//#define RT_ENGINE_IMPLEMENTATION
#ifdef RT_ENGINE_IMPLEMENTATION

#include <device_launch_parameters.h>

namespace RT_ENGINE {

__device__ thread_bvh_metric_reference getThreadBVHMetricReference(){
	int gidx = blockDim.x * blockIdx.x + threadIdx.x;
	int gidy = blockDim.y * blockIdx.y + threadIdx.y;
	int gid = gidy * metrics_width + gidx;

	return thread_bvh_metric_reference{
		g_bvh_metrics.box_tests[gid],
		g_bvh_metrics.triangle_tests[gid],
		g_bvh_metrics.max_depth[gid],
		g_bvh_metrics.branches_encountered[gid]
	};
}

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

	int participants = 0;

	for (int i = 0; i < metrics_length; i++) {
		if (g_bvh_metrics.box_tests[i] == 0) continue;
		participants++;

		total_box_tests += g_bvh_metrics.box_tests[i];
		total_triangle_tests += g_bvh_metrics.triangle_tests[i];
		total_max_depth += g_bvh_metrics.max_depth[i];
		total_branches_encountered += g_bvh_metrics.branches_encountered[i];
	}

	if (participants == 0) throw std::runtime_error("participants is zero.");

	LOG(INFO)
		<< "," << total_box_tests / (float)participants
		<< "," << total_triangle_tests / (float)participants
		<< "," << total_max_depth / (float)participants
		<< "," << total_branches_encountered / (float)participants;

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