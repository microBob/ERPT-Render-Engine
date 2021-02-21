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

using namespace std;

//// SECTION: Class definition
class Raytracing {
private:
	// Context and stream for OptiX to run on
	CUcontext cudaContext{0};
	CUstream cudaStream;


	// OptiX context and pipeline
	OptixDeviceContext optixDeviceContext;

	OptixPipeline optixPipeline;
	OptixPipelineCompileOptions optixPipelineCompileOptions;
	OptixPipelineLinkOptions optixPipelineLinkOptions;


	// OptiX module and programs
	OptixModule optixModule;
	OptixModuleCompileOptions optixModuleCompileOptions;

public:

	void initOptix();

private:

	void createContext();
	void createModules();
};

#endif //ERPT_RENDER_ENGINE_RAYTRACING_H
