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


	/// Setting up Optix pipeline
	// Create system
	createOptixContext();
	createOptixModule();
	// Create program groups
	createRaygenPrograms();
	createMissPrograms();
	createHitgroupPrograms();
	// Create Pipeline and SBT
	createOptiXPipeline();
	createShaderBindingTable();

}

static void contextLogCb(unsigned int level, const char *tag, const char *message, void *) {
	fprintf(stderr, "[%2d][%12s]: %s\n", static_cast<int>(level), tag, message);
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

void Raytracing::createRaygenPrograms() {
	// Set for 1 raygen program group
	raygenProgramGroups.resize(numberOfProgramGroups);

	// Setup specifications for raygen program group
	OptixProgramGroupOptions programGroupOptions = {};
	OptixProgramGroupDesc programGroupDesc = {};
	programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	programGroupDesc.raygen.module = optixModule;
	programGroupDesc.raygen.entryFunctionName = "__raygen__renderFrame";

	// Create raygen program group
	char log[2048];
	size_t logByteSize = sizeof(log);
	auto createOptixRaygenProgramGroup = optixProgramGroupCreate(optixDeviceContext, &programGroupDesc,
	                                                             numberOfProgramGroups,
	                                                             &programGroupOptions, log, &logByteSize,
	                                                             &raygenProgramGroups[0]);
	assert(createOptixRaygenProgramGroup == OPTIX_SUCCESS);
	if (logByteSize > 1) {
		cout << log << endl;
	}
}

void Raytracing::createMissPrograms() {
	// Set for 1 miss program group
	missProgramGroups.resize(numberOfProgramGroups);

	// Setup specifications for miss program groups
	OptixProgramGroupOptions programGroupOptions = {};
	OptixProgramGroupDesc programGroupDesc = {};
	programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	programGroupDesc.miss.module = optixModule;
	programGroupDesc.miss.entryFunctionName = "__miss__radiance";

	// Create miss program group
	char log[2048];
	size_t logByteSize = sizeof(log);
	auto createOptixMissProgramGroup = optixProgramGroupCreate(optixDeviceContext, &programGroupDesc,
	                                                           numberOfProgramGroups,
	                                                           &programGroupOptions, log, &logByteSize,
	                                                           &missProgramGroups[0]);
	assert(createOptixMissProgramGroup == OPTIX_SUCCESS);
	if (logByteSize > 1) {
		cout << log << endl;
	}
}

void Raytracing::createHitgroupPrograms() {
	// Set for 1 miss program group
	hitgroupProgramGroups.resize(numberOfProgramGroups);

	// Setup specifications for miss program groups
	OptixProgramGroupOptions programGroupOptions = {};
	OptixProgramGroupDesc programGroupDesc = {};
	programGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	programGroupDesc.hitgroup.moduleCH = optixModule; // CH = closest hit
	programGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	programGroupDesc.hitgroup.moduleAH = optixModule; // AH = any hit
	programGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	// Create miss program group
	char log[2048];
	size_t logByteSize = sizeof(log);
	auto createOptixHitgroupProgramGroup = optixProgramGroupCreate(optixDeviceContext, &programGroupDesc,
	                                                               numberOfProgramGroups,
	                                                               &programGroupOptions, log, &logByteSize,
	                                                               &hitgroupProgramGroups[0]);
	assert(createOptixHitgroupProgramGroup == OPTIX_SUCCESS);
	if (logByteSize > 1) {
		cout << log << endl;
	}
}

void Raytracing::createOptiXPipeline() {
	// Compile program groups together
	vector<OptixProgramGroup> optixProgramGroups;
	for (auto raygenProgramGroup : raygenProgramGroups) {
		optixProgramGroups.push_back(raygenProgramGroup);
	}
	for (auto missProgramGroup : missProgramGroups) {
		optixProgramGroups.push_back(missProgramGroup);
	}
	for (auto hitgroupProgramGroup : hitgroupProgramGroups) {
		optixProgramGroups.push_back(hitgroupProgramGroup);
	}

	// Create Pipeline
	char log[2048];
	size_t logByteSize = sizeof(log);
	auto createOptixPipeline = optixPipelineCreate(optixDeviceContext, &optixPipelineCompileOptions,
	                                               &optixPipelineLinkOptions,
	                                               optixProgramGroups.data(),
	                                               static_cast<int>(optixProgramGroups.size()), log,
	                                               &logByteSize,
	                                               &optixPipeline);
	assert(createOptixPipeline == OPTIX_SUCCESS);
	if (logByteSize > 1) {
		cout << log << endl;
	}

	// Set pipeline stack size
	auto setOptixPipelineStackSize = optixPipelineSetStackSize(optixPipeline,
	                                                           2 * 1024,
	                                                           2 * 1024,
	                                                           2 * 1024,
	                                                           1);
	assert(setOptixPipelineStackSize == OPTIX_SUCCESS);
	if (logByteSize > 1) {
		cout << log << endl;
	}
}

void Raytracing::createShaderBindingTable() {
	// Raygen records
	vector<RaygenRecord> raygenRecords;
	for (auto raygenProgramGroup : raygenProgramGroups) {
		RaygenRecord record{};
		auto optixSBTRecordPackHeader = optixSbtRecordPackHeader(raygenProgramGroup, &record);
		assert(optixSBTRecordPackHeader == OPTIX_SUCCESS);
		record.data = nullptr; // temporary
		raygenRecords.push_back(record);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	shaderBindingTable.raygenRecord = raygenRecordsBuffer.d_pointer();

	// Miss records
	vector<MissRecord> missRecords;
	for (auto missProgramGroup : missProgramGroups) {
		MissRecord record{};
		auto optixSBTRecordPackHeader = optixSbtRecordPackHeader(missProgramGroup, &record);
		assert(optixSBTRecordPackHeader == OPTIX_SUCCESS);
		record.data = nullptr; // temp
		missRecords.push_back(record);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	shaderBindingTable.missRecordBase = missRecordsBuffer.d_pointer();
	shaderBindingTable.missRecordStrideInBytes = sizeof(MissRecord);
	shaderBindingTable.missRecordCount = static_cast<int>(missRecords.size());

	// Hitgroup records
	int numberOfObjects = 1; // TODO: make this reflect actual number of objects
	vector<HitgroupRecord> hitgroupRecords;
	for (int i = 0; i < numberOfObjects; ++i) {
		int objectType = 0;
		HitgroupRecord record{};
		auto optixSBTRecordPackHeader = optixSbtRecordPackHeader(hitgroupProgramGroups[objectType], &record);
		assert(optixSBTRecordPackHeader == OPTIX_SUCCESS);
		record.objectID = i;
		hitgroupRecords.push_back(record);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	shaderBindingTable.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	shaderBindingTable.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
}

void Raytracing::setFrameSize(const vector2 &newSize) {
	// Update cuda frame buffer
	frameColorBuffer.resize(newSize.x * newSize.y * sizeof(colorVector));

	// Update launch parameters
	optixLaunchParameters.frameBufferSize = newSize;
	optixLaunchParameters.frameColorBuffer = static_cast<colorVector *>(frameColorBuffer.d_ptr);
}

void Raytracing::optixRender() {
	optixLaunchParametersBuffer.upload(&optixLaunchParameters, 1);
	optixLaunchParameters.frameID++;

	auto launchingOptix = optixLaunch(optixPipeline, cudaStream, optixLaunchParametersBuffer.d_pointer(),
	                                  optixLaunchParametersBuffer.sizeInBytes, &shaderBindingTable,
	                                  optixLaunchParameters.frameBufferSize.x, optixLaunchParameters.frameBufferSize.y,
	                                  1);
	assert(launchingOptix == OPTIX_SUCCESS);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
		exit(2);
	}
}
