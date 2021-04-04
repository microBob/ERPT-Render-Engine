//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_RAYTRACING_H
#define ERPT_RENDER_ENGINE_RAYTRACING_H

//// SECTION: Includes and namespace
#include "../include/kernels.cuh"
#include "optix.h"
#include "optix_stubs.h"
#include "CUDABuffer.h"
#include "optixLaunchParameters.h"
#include <vector>
#include <cassert>


//// SECTION: Structs
struct Camera {
	float3 from; // Camera position
	float3 at; // Point camera is looking at
	float3 up; // Global up vector
};
struct TriangleMesh {
	vector<float3> vertices;
	vector<float3> indices;
};

//// SECTION: Class definition
class Raytracing {
public:
	void initOptix(const TriangleMesh &triangleMesh);

	void setFrameSize(const int2 &newSize);

	void optixRender();

	void downloadRender(float *pixData);

	void setCamera(const Camera &camera, const float fov);

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
	OptixTraversableHandle buildAccelerationStructure(const TriangleMesh &triangleMesh);

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

protected:
	Camera lastSetCamera; // Camera used for rendering

	const TriangleMesh mesh; // Mesh definition
	CUDABuffer vertexBuffer;
	CUDABuffer indexBuffer;

	CUDABuffer accelerationStructureBuffer; // Compressed mesh definition

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
	int objectID; // dummy variable
};


#endif //ERPT_RENDER_ENGINE_RAYTRACING_H
