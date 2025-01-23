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


#ifdef RT_ENGINE_IMPLEMENTATION
#include "bvh_metrics.inl"
#endif // ifdef RT_ENGINE_IMPLEMENTATION //

#endif // endif BVH_METRICS_CUDA_H //