//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_KERNELS_CUH
#define ERPT_RENDER_ENGINE_KERNELS_CUH

//// SECTION: Includes
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;


//// SECTION: Cuda meta
class Kernels {
private:
	int gpuID = cudaGetDevice(&gpuID);
	const int cpuID = cudaCpuDeviceId;
public:
	int get_gpuID() const;
	int get_cpuID() const;
};


#endif //ERPT_RENDER_ENGINE_KERNELS_CUH
