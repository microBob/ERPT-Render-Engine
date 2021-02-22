//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_RAYTRACING_H
#define ERPT_RENDER_ENGINE_RAYTRACING_H

//// SECTION: Includes
#include "../include/kernels.cuh"
#include "optix.h"
#include "optix_stubs.h"

#include <cassert>

//// SECTION: Class definition
class Raytracing{
public:
	void initOptix();

private:
	void createOptixContext();

private:
	// CUDA context and stream
	CUcontext cudaContext;
	CUstream cudaStream;

	// OptiX Context
	OptixDeviceContext optixDeviceContext;
};
#endif //ERPT_RENDER_ENGINE_RAYTRACING_H
