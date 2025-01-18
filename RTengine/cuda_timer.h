#ifndef CUDA_TIMER_CLASS_H
#define CUDA_TIMER_CLASS_H

#include <cuda_runtime_api.h>
#include "cuError.h"


class CudaTimer {
	cudaEvent_t start, end;
	float ms = -1.0f;
	bool started;
	bool finished;

	CudaTimer(const CudaTimer&) = delete;
	CudaTimer& operator=(const CudaTimer&) = delete;

public:

	CudaTimer(CudaTimer&& o) {
		start = o.start;
		end = o.end;
		started = o.started;
		finished = o.finished;

		o.start = nullptr;
		o.end = nullptr;
	}
	CudaTimer& operator=(CudaTimer&& o) {
		if (start != nullptr) CUDA_ASSERT(cudaEventDestroy(start));
		if (end != nullptr) CUDA_ASSERT(cudaEventDestroy(end));

		start = o.start;
		end = o.end;
		started = o.started;
		finished = o.finished;

		o.start = nullptr;
		o.end = nullptr;

		return *this;
	}

	CudaTimer() {
		started = false;
		finished = false;
		CUDA_ASSERT(cudaEventCreate(&start));
		CUDA_ASSERT(cudaEventCreate(&end));
	}
	~CudaTimer() {
		if (start != nullptr) CUDA_ASSERT(cudaEventDestroy(start));
		if (end != nullptr) CUDA_ASSERT(cudaEventDestroy(end));
	}

	void Start() {
		assert(started != true);
		CUDA_ASSERT(cudaEventRecord(start));
		started = true;
		finished = false;
	}
	void End() {
		assert(started == true);
		CUDA_ASSERT(cudaEventRecord(end));
		started = false;
		finished = true;
	}

	float ElapsedTime() {
		assert(finished == true);
		if (ms == -1.0f) {
			CUDA_ASSERT(cudaEventSynchronize(end));
			CUDA_ASSERT(cudaEventElapsedTime(&ms, start, end));
		}
		return ms;
	}
};

#endif // ifndef CUDA_TIMER_CLASS_H //