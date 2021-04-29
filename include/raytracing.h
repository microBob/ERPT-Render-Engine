//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_RAYTRACING_H
#define ERPT_RENDER_ENGINE_RAYTRACING_H

//// SECTION: Includes and namespace
#include "../include/kernels.cuh"
#include "optix.h"
#include "optix_stubs.h"
#include "optixLaunchParameters.h"
#include "curand.h"
#include "CUDABuffer.h"
#include <vector>
#include <cassert>


//// SECTION: Structs
struct Camera {
	float3 from; // Camera position
	float3 direction; // Point camera is looking direction
	float3 up; // Up vector
	float cosFovY;
};
struct TriangleMesh {
	vector<float3> vertices;
	vector<uint3> indices;
	colorVector color;
	MeshKind meshKind;
};

//// SECTION: Class definition
class Raytracing {
public:
	void
	initOptix(vector<TriangleMesh> &meshes, size_t numMutations, unsigned long missLimit, unsigned int visibilityTolerance);

	void setFrameSize(const uint2 &newSize);

	void optixRender();

	void downloadRender(float *pixData);

	void setCamera(const Camera &camera);

protected:
	// OptiX base
	void createOptixContext();

	void createOptixModule();

	// Program Groups
	void createRaygenPrograms();

	void createMissPrograms();

	void createHitgroupPrograms();

	// Pipeline and SBT
	void createOptiXPipeline();

	void createShaderBindingTable();

	// Acceleration structure
	OptixTraversableHandle buildAccelerationStructure(vector<TriangleMesh> &meshes);

	// Rendering mutation (random) number
	void generateMutationNumbers(size_t numMutations, unsigned long long int seed, unsigned long missLimit,
	                             unsigned int visibilityTolerance);

private:
	static float3 normalizedVector(float3 vector);

	static float3 vectorCrossProduct(float3 vectorA, float3 vectorB);

protected: // OptiX base
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

	/// Optix ProgramGroups and Shader Binding Table
	short numberOfProgramGroups = 1;
	vector<OptixProgramGroup> raygenProgramGroups;
	CUDABuffer raygenRecordsBuffer;
	vector<OptixProgramGroup> missProgramGroups;
	CUDABuffer missRecordsBuffer;
	vector<OptixProgramGroup> hitgroupProgramGroups;
	CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable shaderBindingTable = {};

protected: // Launch and rendering
	OptixLaunchParameters optixLaunchParameters;
	CUDABuffer optixLaunchParametersBuffer;
	CUDABuffer frameColorBuffer;
	CUDABuffer visibleLocationsBuffer;

protected:
	Camera lastSetCamera; // Camera used for rendering
	vector<TriangleMesh> triangleMeshes; // Mesh definition
	vector<CUDABuffer> vertexBuffer;

	vector<CUDABuffer> indexBuffer;

	CUDABuffer accelerationStructureBuffer; // Compressed triangleMeshes definition
	size_t numberOfMutations;

	CUDABuffer mutationNumbersBuffer;

	CUDABuffer rayHitMetasBuffer;

	CUDABuffer systemStateBuffer;

};

//// SECTION: Shader binding table structs
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void *data; // dummy variable
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void *data; // dummy variable
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	TriangleMeshSBTData data;
};


#endif //ERPT_RENDER_ENGINE_RAYTRACING_H
