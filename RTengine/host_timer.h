#ifndef HOST_TIMER_CLASS_H
#define HOST_TIMER_CLASS_H

#include <ctime>
#include <cstdio>
#include <cassert>


class HostTimer {
	std::clock_t start, end;
	bool started;
	bool finished;

public:

	HostTimer() {
		started = false;
		finished = false;
		start = 0;
		end = 0;
	}

	void Start() {
		assert(started != true);
		start = std::clock();
		started = true;
		finished = false;
	}
	void End() {
		assert(started == true);
		end = std::clock();
		started = false;
		finished = true;
	}

	float ElapsedTimeMS() {
		assert(finished == true);
		return (float)(end - start);
	}
};

#endif // ifndef HOST_TIMER_CLASS_H //