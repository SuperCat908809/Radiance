#ifndef BVH_METRICS_CUDA_INL
#define BVH_METRICS_CUDA_INL

#include "bvh_metrics.h"

#ifdef COLLECT_BVH_METRICS
#include <device_launch_parameters.h>


namespace RT_ENGINE {

__device__ thread_bvh_metric_reference getThreadBVHMetricReference() {
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

	double mean_box_tests = 0.0f;
	double mean_triangle_tests = 0.0f;
	double mean_max_depth = 0.0f;
	double mean_branches_encountered = 0.0f;

	int min_box_tests = INT_MAX;
	int min_triangle_tests = INT_MAX;
	int min_max_depth = INT_MAX;
	int min_branches_encountered = INT_MAX;

	int max_box_tests = 0x0;
	int max_triangle_tests = 0x0;
	int max_max_depth = 0x0;
	int max_branches_encountered = 0x0;

	int participants = 0;

	for (int i = 0; i < metrics_length; i++) {
		if (g_bvh_metrics.box_tests[i] == 0) continue;
		participants++;

		mean_box_tests += g_bvh_metrics.box_tests[i];
		mean_triangle_tests += g_bvh_metrics.triangle_tests[i];
		mean_max_depth += g_bvh_metrics.max_depth[i];
		mean_branches_encountered += g_bvh_metrics.branches_encountered[i];

		if (min_box_tests > g_bvh_metrics.box_tests[i]) min_box_tests = g_bvh_metrics.box_tests[i];
		if (min_triangle_tests > g_bvh_metrics.triangle_tests[i]) min_triangle_tests = g_bvh_metrics.triangle_tests[i];
		if (min_max_depth > g_bvh_metrics.max_depth[i]) min_max_depth = g_bvh_metrics.max_depth[i];
		if (min_branches_encountered > g_bvh_metrics.branches_encountered[i]) min_branches_encountered = g_bvh_metrics.branches_encountered[i];

		if (max_box_tests < g_bvh_metrics.box_tests[i]) max_box_tests = g_bvh_metrics.box_tests[i];
		if (max_triangle_tests < g_bvh_metrics.triangle_tests[i]) max_triangle_tests = g_bvh_metrics.triangle_tests[i];
		if (max_max_depth < g_bvh_metrics.max_depth[i]) max_max_depth = g_bvh_metrics.max_depth[i];
		if (max_branches_encountered < g_bvh_metrics.branches_encountered[i]) max_branches_encountered = g_bvh_metrics.branches_encountered[i];
	}

	mean_box_tests /= participants;
	mean_triangle_tests /= participants;
	mean_max_depth /= participants;
	mean_branches_encountered /= participants;

	double stddev_box_tests = 0.0;
	double stddev_triangle_tests = 0.0;
	double stddev_max_depth = 0.0;
	double stddev_branches_encountered = 0.0;

	for (int i = 0; i < metrics_length; i++) {
		if (g_bvh_metrics.box_tests[i] == 0) continue;

		stddev_box_tests += pow(mean_box_tests - g_bvh_metrics.box_tests[i], 2.0);
		stddev_triangle_tests += pow(mean_triangle_tests - g_bvh_metrics.triangle_tests[i], 2.0);
		stddev_max_depth += pow(mean_max_depth - g_bvh_metrics.max_depth[i], 2.0);
		stddev_branches_encountered += pow(mean_branches_encountered - g_bvh_metrics.branches_encountered[i], 2.0);
	}

	stddev_box_tests /= participants;
	stddev_triangle_tests /= participants;
	stddev_max_depth /= participants;
	stddev_branches_encountered /= participants;

	stddev_box_tests = sqrt(stddev_box_tests);
	stddev_triangle_tests = sqrt(stddev_triangle_tests);
	stddev_max_depth = sqrt(stddev_max_depth);
	stddev_branches_encountered = sqrt(stddev_branches_encountered);

	if (participants == 0) throw std::runtime_error("participants is zero.");

	LOG(INFO)
		<< "," << mean_box_tests //<< "," << stddev_box_tests << "," << min_box_tests << "," << max_box_tests
		<< "," << mean_triangle_tests //<< "," << stddev_triangle_tests << "," << min_triangle_tests << "," << max_triangle_tests
		<< "," << mean_max_depth //<< "," << stddev_max_depth << "," << min_max_depth << "," << max_max_depth
		<< "," << mean_branches_encountered //<< "," << stddev_branches_encountered << "," << min_branches_encountered << "," << max_branches_encountered;
		;
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

#endif // ifdef COLLECT_BVH_METRICS //

#endif // ifndef BVH_METRICS_CUDA_INL //