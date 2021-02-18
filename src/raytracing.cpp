//
// Created by microbobu on 2/15/21.
//
#include "../include/raytracing.h"
#include "optix_function_table_definition.h"

void Raytracing::initOptix() {
	// Reset CUDA and check for GPUs
	cudaFree(nullptr);
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		throw runtime_error("[FATAL ERROR]: No CUDA capable devices found!");
	}

	// Call optixInit and check for errors
	OptixResult init = optixInit();
	if (init != OPTIX_SUCCESS) {
		cerr << "Optix call `optixInit()` failed wit code " << init << " (line " << __LINE__ << ")" << endl;
		exit(2);
	}
}

static void context_log_cb(unsigned int level, const char *tag, const char *message, void *) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

void Raytracing::createContext() {
	// Create stream and context
	cudaStreamCreate(&cudaStream);
	CUresult getCudaContext = cuCtxGetCurrent(&cudaContext);
	assert(getCudaContext == CUDA_SUCCESS);

	// Link to OptiX
	auto createOptixContext = optixDeviceContextCreate(cudaContext, nullptr, &optixDeviceContext);
	assert(createOptixContext == OPTIX_SUCCESS);
	createOptixContext = optixDeviceContextSetLogCallback(optixDeviceContext, context_log_cb, nullptr, 4);
	assert(createOptixContext == OPTIX_SUCCESS);
}
