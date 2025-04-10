#include "renderer.h"

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <easylogging/easylogging++.h>

#include <stb/stb_image_write.h>

#include "cuError.h"
#include "host_timer.h"
#include "cuda_timer.h"

#include "camera.h"
#include "scene.h"

#include "bvh_metrics.h"


namespace RT_ENGINE {
	void _launch_kernel(dim3, dim3, ColorRenderbuffer::handle_cu, Scene::handle_cu, Camera_cu, int);
}

using namespace RT_ENGINE;

Renderer::Renderer(Renderer&& o) noexcept : renderbuffer(std::move(o.renderbuffer)) {}
Renderer& Renderer::operator=(Renderer&& o) noexcept {
	renderbuffer = std::move(o.renderbuffer);
	return *this;
}
Renderer::~Renderer() = default;

Renderer::Renderer(int width, int height) 
	: scene(), renderbuffer(width, height) {}

const ColorRenderbuffer& Renderer::getRenderbuffer() const { return renderbuffer; }

void Renderer::Run(float t) {

	glm::vec3 lookfrom(0, 0, -4);
	lookfrom = glm::rotate(lookfrom, t, glm::vec3(0, 1, 0));
	glm::vec3 lookat(-1.5f, 0, 0);
	glm::vec3 up(0, 1, 0);
	float vfov = glm::radians(36.0f);
	float aspect_ratio = renderbuffer.getWidth() / (float)renderbuffer.getHeight();

	cam = Camera_cu(lookfrom, lookat, up, vfov, aspect_ratio);

	CudaTimer render_kernel_timer{};
	HostTimer render_host_timer{};

	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (renderbuffer.getWidth() + threads.x - 1) / threads.x;
	blocks.y = (renderbuffer.getHeight() + threads.y - 1) / threads.y;
	blocks.z = 1;

	LOG(INFO) << "Renderer::Run ==> Launching render kernel with grid dimensions " << blocks.x << "x" << blocks.y << " : " << threads.x << "x" << threads.y << ".";
	render_kernel_timer.Start();
	render_host_timer.Start();
	//kernel<<<blocks, threads>>>(renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam);
	_launch_kernel(blocks, threads, renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam, 1);
	render_host_timer.End();
	render_kernel_timer.End();
	CUDA_ASSERT(cudaGetLastError());
	CUDA_ASSERT(cudaDeviceSynchronize());

	LOG(INFO) << "Renderer::Run ==> kernel finished in " << render_host_timer.ElapsedTimeMS() << "ms on host and " << render_kernel_timer.ElapsedTimeMS() << "ms on device.";
} // Renderer::Run //

void Renderer::RunFPSTest(int orbit_steps, int frames_per_step, int samples) {

#if TARGET_BVH_ALGORITHM >= WARP_LOCALITY
	dim3 threads = { 8,8,1 };
	dim3 blocks = { (unsigned int)renderbuffer.getWidth(), (unsigned int)renderbuffer.getHeight(), 1 };
#else
	dim3 threads = { 8,8,1 };
	dim3 blocks{};
	blocks.x = (renderbuffer.getWidth() + threads.x - 1) / threads.x;
	blocks.y = (renderbuffer.getHeight() + threads.y - 1) / threads.y;
	blocks.z = 1;
#endif

	CREATE_BVH_METRICS(renderbuffer.getWidth() * renderbuffer.getHeight(), renderbuffer.getWidth());
	RESET_BVH_METRICS;

	LOG(INFO) << "Renderer::RunFPSTest ==> Rendering 5 frames to wake up the device.";

	cam = Camera_cu(
		glm::vec3(0, 2.8f, -4), glm::vec3(0, 2.8f, 0), glm::vec3(0, 1, 0),
		glm::radians(36.0f),
		renderbuffer.getWidth() / (float)renderbuffer.getHeight()
	);

	for (int i = 0; i < 5; i++) {
		_launch_kernel(blocks, threads, renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam, 1);
		CUDA_ASSERT(cudaGetLastError());
		CUDA_ASSERT(cudaDeviceSynchronize());
	}

	LOG(INFO) << "Renderer::RunFPSTest ==> Device properly woken up.";
	LOG(INFO) << "Renderer::RunFPSTest ==> Beginning orbit of model: " << orbit_steps << " orbit steps " << frames_per_step << " frame steps.";

	LOG(INFO) << ",Orbit time,Avg. Frametime,Avg. FPS,Mega-samples per second";

	double average_msps = 0.0;

	for (int orbit_index = 0; orbit_index < orbit_steps; orbit_index++) {

		float time = orbit_index / (float)orbit_steps * glm::radians(720.0f);
		float rotation = glm::radians(orbit_index / (float)orbit_steps * 360.0f);

		glm::vec3 lookfrom = glm::vec3(0, 2.8f, -6);
		//lookfrom = glm::rotate(lookfrom, rotation, glm::vec3(0, 1, 0));
		glm::vec3 lookat = glm::vec3(0, 2.8f, 0);
		glm::vec3 up(0, 1, 0);
		float vfov = glm::radians(45.0f);
		float aspect_ratio = renderbuffer.getWidth() / (float)renderbuffer.getHeight();

		cam = Camera_cu(lookfrom, lookat, up, vfov, aspect_ratio);

		scene.Animate(time);

		RESET_BVH_METRICS;


		float cuda_ms = 0.0f;

		for (int frame_index = 0; frame_index < frames_per_step; frame_index++) {

			CudaTimer render_kernel_timer{};

			render_kernel_timer.Start();
			_launch_kernel(blocks, threads, renderbuffer.getDeviceHandle(), scene.getDeviceHandle(), cam, samples);
			render_kernel_timer.End();

			CUDA_ASSERT(cudaGetLastError());
			CUDA_ASSERT(cudaDeviceSynchronize());

			cuda_ms += render_kernel_timer.ElapsedTimeMS();
		}

		LOG_BVH_METRICS;

#define SAVE_TEST_RENDERS
#ifdef SAVE_TEST_RENDERS
		std::vector<glm::vec3> float_image(renderbuffer.getWidth() * renderbuffer.getHeight());
		renderbuffer.Download(float_image);

		std::vector<glm::u8vec3> image(float_image.size());
		std::transform(float_image.begin(), float_image.end(), image.begin(),
			[](glm::vec3 c) { return c * 255.0f; });

		std::stringstream ss{};
		ss << "out/kernel_bvh_testing_" << std::setw(4) << std::setfill('0') << orbit_index + 1 << ".jpg";
		//ss << "out/kernel_bvh_testing_002.jpg";
		std::string path = ss.str();

		stbi_flip_vertically_on_write(true);
		stbi_write_jpg(path.c_str(), renderbuffer.getWidth(), renderbuffer.getHeight(), 3, image.data(), 90);
#endif // ifdef SAVE_RENDERS //

		//LOG(INFO) << "Renderer::Run ==> kernel finished in " << render_host_timer.ElapsedTimeMS() << "ms on host and " << render_kernel_timer.ElapsedTimeMS() << "ms on device.";
		//LOG(INFO) << "Renderer::Run ==> Orbit index : " << orbit_index + 1 << " avg frametime: " << cuda_ms / frames_per_step << "ms, avg FPS: " << 1000 * frames_per_step / cuda_ms << ".";

		//LOG(INFO) << "," << orbit_index + 1 << "," << cuda_ms / frames_per_step << "," << 1000 * frames_per_step / cuda_ms;

#if TARGET_BVH_ALGORITHM < WARP_LOCALITY
		int total_samples = renderbuffer.getWidth() * renderbuffer.getHeight() * samples;
#else
		int total_samples = renderbuffer.getWidth() * renderbuffer.getHeight() * samples * threads.x * threads.y;
#endif
		float seconds_per_kernel = cuda_ms * 1e-3f / frames_per_step;
		float samples_per_second = total_samples / seconds_per_kernel;

		LOG(INFO)
			<< "," << orbit_index + 1
			<< "," << cuda_ms / frames_per_step
			<< "," << 1000 * frames_per_step / cuda_ms
			<< "," << samples_per_second / 1e6f;

		average_msps += samples_per_second / 1e6f;
	}

	average_msps /= orbit_steps;
	LOG(INFO) << "Average Mega-samples per second: " << average_msps;

	DELETE_BVH_METRICS;
} // Renderer::Run //