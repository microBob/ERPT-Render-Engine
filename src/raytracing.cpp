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

	// Create OptiX context from CUDA context
	createContext();
}

static void contextLogCb(unsigned int level, const char *tag, const char *message, void *) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

void Raytracing::createContext() {
	// Create stream and context
	cudaSetDevice(k.get_gpuID());
	cudaStreamCreate(&cudaStream);
	CUresult getCudaContext = cuCtxGetCurrent(&cudaContext);
	assert(getCudaContext == CUDA_SUCCESS);

	// Link to OptiX
	auto createOptixContext = optixDeviceContextCreate(cudaContext, nullptr, &optixDeviceContext);
	assert(createOptixContext == OPTIX_SUCCESS);
	createOptixContext = optixDeviceContextSetLogCallback(optixDeviceContext, contextLogCb, nullptr, 4);
	assert(createOptixContext == OPTIX_SUCCESS);
}

extern "C" char embeddedPtxCode[];

void Raytracing::createModules() {
	optixModuleCompileOptions.maxRegisterCount = 50;
	optixModuleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT; // Optimization level
	optixModuleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // No debug

	optixPipelineCompileOptions = {};
	optixPipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	optixPipelineCompileOptions.usesMotionBlur = false;
	optixPipelineCompileOptions.numPayloadValues = 2;
	optixPipelineCompileOptions.numAttributeValues = 2;
	optixPipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	optixPipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

	optixPipelineLinkOptions.maxTraceDepth = 2;

	const string ptxCode = embeddedPtxCode;

	char log[2048];
	size_t sizeofLog = sizeof(log);
	auto moduleCreate = optixModuleCreateFromPTX(optixDeviceContext, &optixModuleCompileOptions,
	                                             &optixPipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log,
	                                             &sizeofLog, &optixModule);
	assert(moduleCreate == OPTIX_SUCCESS);
	if (sizeofLog > 1) {
		cout << log << endl;
	}
}
