//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_RAYTRACING_H
#define ERPT_RENDER_ENGINE_RAYTRACING_H

//// SECTION: Includes and namespace
#include "../include/kernels.cuh"
#include "optix.h"
#include "optix_stubs.h"
#include <vector>

#include <cassert>

//// SECTION: Class definition
class Raytracing {
public:
	void initOptix();

protected:
	void createOptixContext();

	void createOptixModule();

protected:
	// CUDA context and stream
	CUcontext cudaContext;
	CUstream cudaStream;

	// OptiX Context
	OptixDeviceContext optixDeviceContext;

	// OptiX Pipeline
	OptixPipeline optixPipeline;
	OptixPipelineCompileOptions optixPipelineCompileOptions = {};
	OptixPipelineLinkOptions optixPipelineLinkOptions = {};

	// OptiX Module
	OptixModule optixModule;
	OptixModuleCompileOptions optixModuleCompileOptions = {};

	// Embedded PTX device code

};

#endif //ERPT_RENDER_ENGINE_RAYTRACING_H
