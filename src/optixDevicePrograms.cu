#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"
//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

/// Launch Parameters
extern "C" __constant__ OptixLaunchParameters optixLaunchParameters;

enum {
	SURFACE_RAY_TYPE = 0,
	RAY_TYPE_COUNT
};

/// Utility functions
__device__ float3 normalizeVectorGPU(float3 vector) {
	const auto r_normal = rnorm3df(vector.x, vector.y, vector.z);

	return make_float3(vector.x * r_normal, vector.y * r_normal, vector.z * r_normal);
}

__device__ float3 vectorCrossProductGPU(float3 vectorA, float3 vectorB) {
	return make_float3(vectorA.y * vectorB.z - vectorA.z * vectorB.y, vectorA.z * vectorB.x - vectorA.x * vectorB.z,
	                   vectorA.x * vectorB.y - vectorA.y * vectorB.x);
}

__device__ float vectorDotProductGPU(float3 vectorA, float3 vectorB) {
	return vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z;
}

/// Payload management
static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t rawPointer = static_cast<uint64_t>(i0) << 32 | i1;
	void *pointer = reinterpret_cast<void *>(rawPointer);
	return pointer;
}

static __forceinline__ __device__ void packPointer(void *pointer, uint32_t &i0, uint32_t &i1) {
	const auto rawPointer = reinterpret_cast<uint64_t>(pointer);
	i0 = rawPointer >> 32;
	i1 = rawPointer & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPerRayData() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T *>( unpackPointer(u0, u1));
}

/// Ray generation program
extern "C" __global__ void __raygen__renderFrame() {
	// Get index and camera
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;
	const unsigned int mutationNumberIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	unsigned int screenX = llrintf(static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x) *
	                               optixLaunchParameters.curMutationNumbers[mutationNumberIndex]);
	unsigned int screenY = llrintf(static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y) *
	                               optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1]);
	unsigned int pixelIndex = screenX + screenY * optixLaunchParameters.frame.frameBufferSize.x;

	const auto &camera = optixLaunchParameters.camera;

	/// Starting ray from camera
	// Create per ray data
	PerRayData rayData;
	uint32_t payload0, payload1;
	packPointer(&rayData, payload0, payload1);

	// Create base screen ray
	const auto screen = make_float2(
		(static_cast<float>(screenX) + 0.5f) /
		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
		(static_cast<float>(screenY) + 0.5f) /
		static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y));
	auto screenMinus = make_float2(screen.x - 0.5f, screen.y - 0.5f);
	auto horizontalTimesScreenMinus = make_float3(screenMinus.x * camera.horizontal.x,
	                                              screenMinus.x * camera.horizontal.y,
	                                              screenMinus.x * camera.horizontal.z);
	auto verticalTimesScreenMinus = make_float3(screenMinus.y * camera.vertical.x,
	                                            screenMinus.y * camera.vertical.y,
	                                            screenMinus.y * camera.vertical.z);
	auto rawRayDirection = make_float3(
		camera.direction.x + horizontalTimesScreenMinus.x + verticalTimesScreenMinus.x,
		camera.direction.y + horizontalTimesScreenMinus.y + verticalTimesScreenMinus.y,
		camera.direction.z + horizontalTimesScreenMinus.z + verticalTimesScreenMinus.z);

	float3 rayOrigin = camera.position;
	float3 rayDirectionNormalized = normalizeVectorGPU(rawRayDirection);
	atomicAdd(&optixLaunchParameters.pixelVisits[pixelIndex], 1);

	// Trace
	optixTrace(optixLaunchParameters.optixTraversableHandle,
	           rayOrigin,
	           rayDirectionNormalized,
	           0.001f, // Needs to have gone somewhere
	           1e20f,
	           0.0f,
	           OptixVisibilityMask(255),
	           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
	           SURFACE_RAY_TYPE,
	           RAY_TYPE_COUNT,
	           SURFACE_RAY_TYPE,
	           payload0,
	           payload1);

	colorVector baseColor;
	bool raySuccessful;
	if (rayData.normal.x + rayData.normal.y + rayData.normal.z != 0) {
		baseColor = rayData.color;

		// Increment Energy at pixel if a light source was hit
		if (rayData.light) {
			raySuccessful = true;
		} else { // Else, continue with second ray
			/// Reflected ray
			for (int depthIndex = 0; depthIndex < optixLaunchParameters.traceDepth; ++depthIndex) {
				// Create ray
				const float r = sqrt(
					optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2 + depthIndex * 2]);
				const float phi = 2 * 3.1415f *
				                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 3 + depthIndex * 2];
				const float circleX = r * cos(phi);
				const float circleY = r * sin(phi);
				const float circleZ = sqrt(1 - (r * r));
				const float3 newDirection = make_float3(
					rayData.xAxis.x * circleX + rayData.yAxis.x * circleY + rayData.normal.x * circleZ,
					rayData.xAxis.y * circleX + rayData.yAxis.y * circleY + rayData.normal.y * circleZ,
					rayData.xAxis.z * circleX + rayData.yAxis.z * circleY + rayData.normal.z * circleZ);

				rayOrigin = rayData.location;
				rayDirectionNormalized = newDirection;//normalizeVectorGPU(newDirection);


				// Trace
				optixTrace(optixLaunchParameters.optixTraversableHandle,
				           rayOrigin,
				           rayDirectionNormalized,
				           0.001f, // Needs to have gone somewhere
				           1e20f,
				           0.0f,
				           OptixVisibilityMask(255),
				           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				           SURFACE_RAY_TYPE,
				           RAY_TYPE_COUNT,
				           SURFACE_RAY_TYPE,
				           payload0,
				           payload1);

				// Stop if there was a miss
				if (rayData.normal.x + rayData.normal.y + rayData.normal.z == 0) {
					break;
				}
				// If there's light, increment data
				if (rayData.light) {
					raySuccessful = true;
					atomicAdd(&optixLaunchParameters.energyPerPixel[pixelIndex], rayData.energy);
					break;
				}
			}
		}

		if (raySuccessful) {
			const float colorSum =
				(baseColor.r + baseColor.g + baseColor.b) / rayData.energy *
				static_cast<float>(optixLaunchParameters.samples.total);
			atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].r, baseColor.r / colorSum);
			atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].g, baseColor.g / colorSum);
			atomicAdd(&optixLaunchParameters.frame.frameColorBuffer[pixelIndex].b, baseColor.b / colorSum);
		}
	}
}

/// Miss program
extern "C" __global__ void __miss__radiance() {
	PerRayData &perRayData = *(PerRayData *) getPerRayData<PerRayData>();
	const auto zeroVector = make_float3(0, 0, 0);
	perRayData = {zeroVector, zeroVector, zeroVector, zeroVector, {}, 0, false};
}

/// Hit program
extern "C" __global__ void __closesthit__radiance() {
	const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();

	// Essential hit data
	const float3 rayDir = optixGetWorldRayDirection();
	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float rayLength = optixGetRayTmax();

	// Check if valid hit
	if (rayLength <= 0.0001) { // Basically 0
		printf("No ray length, skipping\n");
		return;
	}

	const float3 hitLocation = make_float3(rayOrigin.x + rayLength * rayDir.x, rayOrigin.y + rayLength * rayDir.y,
	                                       rayOrigin.z + rayLength * rayDir.z);

	// Surface normal
	const int primitiveIndex = optixGetPrimitiveIndex();
	const int3 index = sbtData.index[primitiveIndex];
	const float3 &vertexA = sbtData.vertex[index.x];
	const float3 &vertexB = sbtData.vertex[index.y];
	const float3 &vertexC = sbtData.vertex[index.z];
	const auto vertexBMinusA = make_float3(vertexB.x - vertexA.x, vertexB.y - vertexA.y, vertexB.z - vertexA.z);
	const auto vertexCMinusA = make_float3(vertexC.x - vertexA.x, vertexC.y - vertexA.y, vertexC.z - vertexA.z);
	const float3 normalAxis = normalizeVectorGPU(vectorCrossProductGPU(vertexBMinusA, vertexCMinusA));
	const colorVector normalColor = {(normalAxis.x + 1) / 2, (normalAxis.y + 1) / 2, (normalAxis.z + 1) / 2};

	// Second Axis
	// TODO: causing bad indirect lighting
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;
	const unsigned int mutationNumberIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	const float3 yAxis = normalizeVectorGPU(
		vectorCrossProductGPU(make_float3(optixLaunchParameters.curMutationNumbers[mutationNumberIndex],
		                                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 1],
		                                  optixLaunchParameters.curMutationNumbers[mutationNumberIndex + 2]),
		                      normalAxis));
//	const float3 yAxis = normalizeVectorGPU(vectorCrossProductGPU(optixLaunchParameters.camera.direction, normalAxis));

	// Third Axis
	const float3 xAxis = normalizeVectorGPU(vectorCrossProductGPU(normalAxis, yAxis));

	// Encode per ray data
	PerRayData &perRayData = *(PerRayData *) getPerRayData<PerRayData>();
	perRayData = {hitLocation, normalAxis, xAxis, yAxis, sbtData.color, sbtData.energy, sbtData.kind == Light};
}
extern "C" __global__ void __anyhit__radiance() {}

#pragma clang diagnostic pop