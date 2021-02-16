//
// Created by microbobu on 2/15/21.
//
#include "../include/raytracing.h"
#include "optix_function_table_definition.h"

void Raytracing::initOptix() {
	cudaFree(nullptr);
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		throw runtime_error("[FATAL ERROR]: No CUDA capable devices found!");
	}

	OptixResult init = optixInit();
	if (init != OPTIX_SUCCESS) {
		cerr << "Optix call `optixInit()` failed wit code " << init << " (line " << __LINE__ << ")" << endl;
		exit(2);
	}
}
