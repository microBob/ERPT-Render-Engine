//
// Created by microbobu on 2/15/21.
//
#include "../include/raytracing.h"
#include "optix_function_table_definition.h"

void Raytracing::initOptix() {
	/// Initialize Optix library
	// Reset and prep CUDA
	cudaFree(nullptr);
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		throw runtime_error("[FATAL ERROR]: No CUDA capable devices found!");
	}

	// Call optixInit
	OptixResult init = optixInit();
	if (init != OPTIX_SUCCESS) {
		cerr << "Optix call `optixInit()` failed wit code " << init << " (line " << __LINE__ << ")" << endl;
		exit(2);
	}


	/// Create Optix context
	createOptixContext();


}

static void contextLogCb(unsigned int level, const char *tag, const char *message, void *) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

void Raytracing::createOptixContext() {
	cudaSetDevice(k.get_gpuID());
	cudaStreamCreate(&cudaStream);

	auto getContext = cuCtxGetCurrent(&cudaContext);
	assert(getContext == CUDA_SUCCESS);

	auto createOptixDeviceContext = optixDeviceContextCreate(cudaContext, nullptr, &optixDeviceContext);
	assert(createOptixDeviceContext == OPTIX_SUCCESS);
	auto setOptixCallbackLogLevel = optixDeviceContextSetLogCallback(optixDeviceContext, contextLogCb, nullptr, 4);
	assert(setOptixCallbackLogLevel == OPTIX_SUCCESS);
}

extern "C" char embeddedPtxCode[];

void Raytracing::createOptixModule() {
	// Module compile options
	optixModuleCompileOptions.maxRegisterCount = 50;
	optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	// Pipeline compile options
	optixPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	optixPipelineCompileOptions.usesMotionBlur = false;
	optixPipelineCompileOptions.numPayloadValues = 2;
	optixPipelineCompileOptions.numAttributeValues = 2;
	optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	optixPipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParameters";

	// Pipeline linking options
	optixPipelineLinkOptions.maxTraceDepth = 2;

	// Create module
	string ptxCode = embeddedPtxCode;
	char log[2048];
	size_t logByteSize = sizeof(log);
	auto createOptixModuleFromPTX = optixModuleCreateFromPTX(optixDeviceContext, &optixModuleCompileOptions,
	                                                         &optixPipelineCompileOptions, ptxCode.c_str(),
	                                                         ptxCode.size(),
	                                                         log, &logByteSize, &optixModule);
	assert(createOptixModuleFromPTX == OPTIX_SUCCESS);

	if (logByteSize > 1) {
		cout << log << endl;
	}
}
