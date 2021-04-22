#include "../include/raytracing.h"
#include "optix_function_table_definition.h"

//
// Created by microbobu on 2/15/21.
//

void Raytracing::initOptix(TriangleMesh &newMesh) {
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
		cerr << "Optix call `optixInit()` failed with code " << init << " (line " << __LINE__ << ")" << endl;
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
	// Create Acceleration Structure
	triangleMesh = newMesh;
	optixLaunchParameters.optixTraversableHandle = buildAccelerationStructure(newMesh);
	// Generate mutation numbers
	generateMutationNumbers(3 * 100);
	// Create Pipeline and SBT
	createOptiXPipeline();
	createShaderBindingTable();

	optixLaunchParametersBuffer.alloc(sizeof(optixLaunchParameters));
}

extern "C" char embeddedPtxCode[];

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
	auto setOptixCallbackLogLevel = optixDeviceContextSetLogCallback(optixDeviceContext, contextLogCb, nullptr, 2);
	assert(setOptixCallbackLogLevel == OPTIX_SUCCESS);
}

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
//		cout << log << endl;
	}

	// Set pipeline stack size
	auto setOptixPipelineStackSize = optixPipelineSetStackSize(optixPipeline,
	                                                           2 * 1024,
	                                                           2 * 1024,
	                                                           2 * 1024,
	                                                           1);
	assert(setOptixPipelineStackSize == OPTIX_SUCCESS);
	if (logByteSize > 1) {
//		cout << log << endl;
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
		HitgroupRecord record;
		auto optixSBTRecordPackHeader = optixSbtRecordPackHeader(hitgroupProgramGroups[objectType], &record);
		assert(optixSBTRecordPackHeader == OPTIX_SUCCESS);
		record.data.vertex = (float3 *) vertexBuffer.d_pointer();
		record.data.index = (int3 *) indexBuffer.d_pointer();
		record.data.color = triangleMesh.color;
		hitgroupRecords.push_back(record);
	}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	shaderBindingTable.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	shaderBindingTable.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	shaderBindingTable.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
}

void Raytracing::setFrameSize(const uint2 &newSize) {
	// Update cuda frame buffer
	frameColorBuffer.resize(newSize.x * newSize.y * sizeof(colorVector));

	// Update launch parameters
	optixLaunchParameters.frame.frameBufferSize = newSize;
	optixLaunchParameters.frame.frameColorBuffer = static_cast<colorVector *>(frameColorBuffer.d_ptr);
}

void Raytracing::optixRender() {
	optixLaunchParametersBuffer.upload(&optixLaunchParameters, 1);

	auto launchingOptix = optixLaunch(optixPipeline, cudaStream, optixLaunchParametersBuffer.d_pointer(),
	                                  optixLaunchParametersBuffer.sizeInBytes, &shaderBindingTable,
	                                  optixLaunchParameters.frame.frameBufferSize.x,
	                                  optixLaunchParameters.frame.frameBufferSize.y,
	                                  1);
	assert(launchingOptix == OPTIX_SUCCESS);

	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
		exit(2);
	}
}

void Raytracing::downloadRender(float *pixData) {
	unsigned int numberOfPixels =
		optixLaunchParameters.frame.frameBufferSize.x * optixLaunchParameters.frame.frameBufferSize.y;
	// Copy back rendered pixels as colorVectors
	auto *renderedPixelVectors = static_cast<colorVector *>(malloc(numberOfPixels * sizeof(colorVector)));
	frameColorBuffer.download(renderedPixelVectors, numberOfPixels);

	// Copy data into pixData
	for (int i = 0; i < numberOfPixels; ++i) {
		pixData[i * 4] = renderedPixelVectors[i].r;
		pixData[i * 4 + 1] = renderedPixelVectors[i].g;
		pixData[i * 4 + 2] = renderedPixelVectors[i].b;
		pixData[i * 4 + 3] = renderedPixelVectors[i].a;
	}
}

void Raytracing::setCamera(const Camera &camera) {
	lastSetCamera = camera;
	optixLaunchParameters.camera.position = camera.from;
	optixLaunchParameters.camera.direction = camera.direction;

	const float aspectRatioFov = camera.cosFovY * static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x) /
	                             static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y);

	float3 normalizedHorizontalCross = normalizedVector(
		vectorCrossProduct(optixLaunchParameters.camera.direction, camera.up));
	optixLaunchParameters.camera.horizontal = make_float3(aspectRatioFov * normalizedHorizontalCross.x,
	                                                      aspectRatioFov * normalizedHorizontalCross.y,
	                                                      aspectRatioFov * normalizedHorizontalCross.z);
	float3 normalizedVerticalCross = normalizedVector(
		vectorCrossProduct(optixLaunchParameters.camera.horizontal, optixLaunchParameters.camera.direction));
	optixLaunchParameters.camera.vertical = make_float3(camera.cosFovY * normalizedVerticalCross.x,
	                                                    camera.cosFovY * normalizedVerticalCross.y,
	                                                    camera.cosFovY * normalizedVerticalCross.z);
}

OptixTraversableHandle Raytracing::buildAccelerationStructure(TriangleMesh &triMesh) {
	// Upload model to GPU
	vertexBuffer.alloc_and_upload(triMesh.vertices);
	indexBuffer.alloc_and_upload(triMesh.indices);

	OptixTraversableHandle accelerationStructureHandle{0};

	/// Triangle Inputs
	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	// Local pointers to data
	CUdeviceptr deviceVertices = vertexBuffer.d_pointer();
	CUdeviceptr deviceIndices = indexBuffer.d_pointer();

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = static_cast<int>(triMesh.vertices.size());
	triangleInput.triangleArray.vertexBuffers = &deviceVertices;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);
	triangleInput.triangleArray.numIndexTriplets = static_cast<int>(triMesh.indices.size());
	triangleInput.triangleArray.indexBuffer = deviceIndices;

	uint32_t triangleInputFlags[1] = {0};

	// SBT records
	triangleInput.triangleArray.flags = triangleInputFlags;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

	/// BLAS setup
	OptixAccelBuildOptions accelBuildOptions = {};
	accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelBuildOptions.motionOptions.numKeys = 1;
	accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	auto accelComputerMemoryUsage = optixAccelComputeMemoryUsage(optixDeviceContext, &accelBuildOptions, &triangleInput,
	                                                             1, &blasBufferSizes);
	assert(accelComputerMemoryUsage == OPTIX_SUCCESS);

	/// Prep compaction
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	/// Build
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

	auto accelBuild = optixAccelBuild(optixDeviceContext, nullptr, &accelBuildOptions, &triangleInput, 1,
	                                  tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(),
	                                  outputBuffer.sizeInBytes, &accelerationStructureHandle, &emitDesc, 1);
	cudaDeviceSynchronize();
	assert(accelBuild == OPTIX_SUCCESS);

	/// Compaction
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	accelerationStructureBuffer.alloc(compactedSize);
	auto accelCompact = optixAccelCompact(optixDeviceContext, nullptr, accelerationStructureHandle,
	                                      accelerationStructureBuffer.d_pointer(),
	                                      accelerationStructureBuffer.sizeInBytes, &accelerationStructureHandle);
	cudaDeviceSynchronize();
	assert(accelCompact == OPTIX_SUCCESS);

	/// Cleanup
	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();

	return accelerationStructureHandle;
}

void Raytracing::generateMutationNumbers(size_t nFloats) {
	// Setup
	curandGenerator_t gen;
	float *deviceNumbers, *hostNumbers;
	size_t arraySize = nFloats * sizeof(float);

	// Allocate memory for data
	hostNumbers = static_cast<float *>(calloc(nFloats, sizeof(float)));
	cudaMalloc(reinterpret_cast<void **>(&deviceNumbers), arraySize);

	// cuRAND setup
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 0);
	curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_SEEDED);

	// Generate
	curandGenerateUniform(gen, deviceNumbers, nFloats);

	// Copy back to host
	cudaMemcpy(hostNumbers, deviceNumbers, arraySize, cudaMemcpyDeviceToHost);

	// Re-upload as buffer; TODO: improve this to not use re-upload
	vector<float> vectorizedArray(hostNumbers, hostNumbers + nFloats);
	mutationNumbersBuffer.alloc_and_upload(vectorizedArray);
	optixLaunchParameters.mutationNumbers = (float *) mutationNumbersBuffer.d_pointer();

	// Also create ray hit meta buffer
	RayHitMeta *hostHitMetas;
	hostHitMetas = static_cast<RayHitMeta *>(calloc(nFloats / 3, sizeof(RayHitMeta)));
	vector<RayHitMeta> vectorizedHitMetaArray(hostHitMetas, hostHitMetas + nFloats);
	rayHitMetasBuffer.alloc_and_upload(vectorizedHitMetaArray);
	optixLaunchParameters.rayHitMetas = (RayHitMeta *) rayHitMetasBuffer.d_pointer();
}

float3 Raytracing::normalizedVector(float3 vector) {
	auto magnitude = static_cast<float>(sqrt(pow(vector.x, 2) + pow(vector.y, 2) + pow(vector.z, 2)));

	return make_float3(vector.x / magnitude, vector.y / magnitude, vector.z / magnitude);
}

float3 Raytracing::vectorCrossProduct(float3 vectorA, float3 vectorB) {
	return make_float3(vectorA.y * vectorB.z - vectorA.z * vectorB.y, vectorA.z * vectorB.x - vectorA.x * vectorB.z,
	                   vectorA.x * vectorB.y - vectorA.y * vectorB.x);
}
