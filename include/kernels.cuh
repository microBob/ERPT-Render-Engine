//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_KERNELS_CUH
#define ERPT_RENDER_ENGINE_KERNELS_CUH

//// SECTION: Includes
#include <iostream>
#include <cassert>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;


//// SECTION: Cuda meta
class Kernels {
private:
	cudaDeviceProp prop{};

	int gpuID = cudaGetDevice(&gpuID);
	const int cpuID = cudaCpuDeviceId;

	int threadsToLaunchForVertices;
	int blocksToLaunchForVertices;
public:
	Kernels();

	int get_gpuID() const {
		return gpuID;
	};

	int get_cpuID() const {
		return cpuID;
	};

	int get_threadsToLaunchForVertices() const {
		return threadsToLaunchForVertices;
	}

	int get_blocksToLaunchForVertices() const {
		return blocksToLaunchForVertices;
	}

	void set_kernelThreadsAndBlocks(int sceneVertexCount);
};

static Kernels k;


#endif //ERPT_RENDER_ENGINE_KERNELS_CUH
