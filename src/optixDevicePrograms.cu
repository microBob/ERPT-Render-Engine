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
	auto r_normal = rnorm3df(vector.x, vector.y, vector.z);

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
	const auto &camera = optixLaunchParameters.camera;

	// Ray information storage
	float3 rayOrigin; // Where the ray starts
	float3 rayDirectionNormalized; // Where the ray goes

	if (optixLaunchParameters.systemState[MutationIndex] <
	    optixLaunchParameters.mutation.numberOfThem) { // Still have mutations = rendering
		/// Generate random ray direction
		// Get random number set
		const float3 &mutationNumbersSet = optixLaunchParameters.mutation.numbers[optixLaunchParameters.systemState[MutationIndex]];
		if (optixLaunchParameters.systemState[StartFromScreenBool]) {
			const unsigned int randScreenX = llrintf(
				mutationNumbersSet.x * static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x - 1));
			const unsigned int randScreenY = llrintf(
				mutationNumbersSet.y * static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y - 1));
			const auto screen = make_float2(
				(static_cast<float>(randScreenX) + 0.5f) /
				static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
				(static_cast<float>(randScreenY) + 0.5f) /
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

			rayOrigin = camera.position;
			rayDirectionNormalized = normalizeVectorGPU(rawRayDirection);
		} else {
			const float3 newRayDirRaw = make_float3(cospif(mutationNumbersSet.x), cospif(mutationNumbersSet.y),
			                                        cospif(mutationNumbersSet.z));
			const float rayDirInverseMagnitude = rnorm3df(newRayDirRaw.x, newRayDirRaw.y, newRayDirRaw.z);
			RayHitMeta sourceRayMeta = optixLaunchParameters.rayHit.metas[optixLaunchParameters.systemState[RayHitMetaIndex]];

			rayOrigin = sourceRayMeta.hitLocation;
			rayDirectionNormalized = make_float3(newRayDirRaw.x * rayDirInverseMagnitude + sourceRayMeta.hitNormal.x,
			                                     newRayDirRaw.y * rayDirInverseMagnitude + sourceRayMeta.hitNormal.y,
			                                     newRayDirRaw.z * rayDirInverseMagnitude + sourceRayMeta.hitNormal.z);
		}
		optixLaunchParameters.systemState[MutationIndex]++;

		// Optix Trace
		optixTrace(optixLaunchParameters.optixTraversableHandle,
		           rayOrigin,
		           rayDirectionNormalized,
		           0.f,
		           1e20f,
		           0.0f,
		           OptixVisibilityMask(255),
		           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		           SURFACE_RAY_TYPE,
		           RAY_TYPE_COUNT,
		           SURFACE_RAY_TYPE);
	} else { // Done rendering and is now checking for visibility
		// Create per ray data pointer
		colorVector pixelColorPerRayData;
		uint32_t payload0, payload1;
		packPointer(&pixelColorPerRayData, payload0, payload1);

		// Creating screen ray
		const auto screen = make_float2(
			(static_cast<float>(ix) + 0.5f) /
			static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
			(static_cast<float>(iy) + 0.5f) /
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

		rayOrigin = camera.position;
		rayDirectionNormalized = normalizeVectorGPU(rawRayDirection);

		// Do trace
		optixTrace(optixLaunchParameters.optixTraversableHandle,
		           rayOrigin,
		           rayDirectionNormalized,
		           0.f,
		           1e20f,
		           0.0f,
		           OptixVisibilityMask(255),
		           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		           SURFACE_RAY_TYPE,
		           RAY_TYPE_COUNT,
		           SURFACE_RAY_TYPE,
		           payload0,
		           payload1);

		// Loop through recorded hits
		const unsigned int frameIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
		float3 visibilityHitLocation = optixLaunchParameters.frame.visibleLocations[frameIndex];
		float energy = 0;

		if (visibilityHitLocation.x != nanf("")) {
			for (unsigned int hitIndex = 0; hitIndex <= optixLaunchParameters.systemState[RayHitMetaIndex]; ++hitIndex) {
				RayHitMeta thisHitMeta = optixLaunchParameters.rayHit.metas[hitIndex];
				float3 rayHitLocation = thisHitMeta.hitLocation;
				float visibilityTolerance = optixLaunchParameters.rayHit.visibilityTolerance;
				bool inXRange = fdimf(visibilityHitLocation.x, rayHitLocation.x) < visibilityTolerance;
				bool inYRange = fdimf(visibilityHitLocation.y, rayHitLocation.y) < visibilityTolerance;
				bool inZRange = fdimf(visibilityHitLocation.z, rayHitLocation.z) < visibilityTolerance;

				if (inXRange && inYRange && inZRange) {
					energy = thisHitMeta.energy;
					break;
				}
			}
		}

		// Edit pixelColorPerRayData and record
		pixelColorPerRayData = {pixelColorPerRayData.r * energy, pixelColorPerRayData.g * energy,
		                        pixelColorPerRayData.b * energy};
		optixLaunchParameters.frame.frameColorBuffer[frameIndex] = pixelColorPerRayData;
	}
}

/// Miss program
extern "C" __global__ void __miss__radiance() {
	if (optixLaunchParameters.systemState[MutationIndex] ==
	    optixLaunchParameters.mutation.numberOfThem) { // Visibility check operation
		const unsigned int ix = optixGetLaunchIndex().x;
		const unsigned int iy = optixGetLaunchIndex().y;
		const unsigned int visibleIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
		optixLaunchParameters.frame.visibleLocations[visibleIndex] = make_float3(nanf(""), nanf(""), nanf(""));
	}
}

/// Hit program
extern "C" __global__ void __closesthit__radiance() {
	const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *) optixGetSbtDataPointer();
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;

	// Essential hit data
	const float3 rayDir = optixGetWorldRayDirection();
	const float3 rayOrigin = optixGetWorldRayOrigin();
	const float rayLength = optixGetRayTmax();
	const float3 hitLocation = make_float3(rayOrigin.x + rayLength * rayDir.x, rayOrigin.y + rayLength * rayDir.y,
	                                       rayOrigin.z + rayLength * rayDir.z);

	if (optixLaunchParameters.systemState[MutationIndex] < optixLaunchParameters.mutation.numberOfThem) { // Trace operation
		if (sbtData.kind == Mesh) {
			// Compute surface normal
			const int primitiveIndex = optixGetPrimitiveIndex();
			const int3 index = sbtData.index[primitiveIndex];
			const float3 &vertexA = sbtData.vertex[index.x];
			const float3 &vertexB = sbtData.vertex[index.y];
			const float3 &vertexC = sbtData.vertex[index.z];

			auto vertexBMinusA = make_float3(vertexB.x - vertexA.x, vertexB.y - vertexA.y, vertexB.z - vertexA.z);
			auto vertexCMinusA = make_float3(vertexC.x - vertexA.x, vertexC.y - vertexA.y, vertexC.z - vertexA.z);
			const float3 Ng = normalizeVectorGPU(vectorCrossProductGPU(vertexBMinusA, vertexCMinusA));

			// "NdotD" shading
			const float cosDN = 0.2f + 0.8f * fabsf(vectorDotProductGPU(rayDir, Ng));

			// Create hit meta
			RayHitMeta thisHitMeta = {hitLocation, rayOrigin, Ng, rayLength, 1,
			                          optixLaunchParameters.systemState[StartFromScreenBool] == 1, optixLaunchParameters.systemState[1],
			                          cosDN};
			optixLaunchParameters.systemState[RayHitMetaIndex]++;
			optixLaunchParameters.rayHit.metas[optixLaunchParameters.systemState[RayHitMetaIndex]] = thisHitMeta;
			if (optixLaunchParameters.systemState[StartFromScreenBool]) {
				optixLaunchParameters.systemState[StartFromScreenBool] = 0;
			}

			// Debug prints
//			printf("Hit for 480, 270\n");// Calculate hit hitLocation
//			printf("Ray Origin:\t\t%f, %f, %f\n", rayOrigin.x, rayOrigin.y, rayOrigin.z);
//			printf("Ray Direction:\t%f, %f, %f\n", rayDir.x, rayDir.y, rayDir.z);
//			printf("Ray Length:\t\t%f\n", rayLength);
//			printf("Hit Location:\t%f, %f, %f\n", hitLocation.x, hitLocation.y,
//			       hitLocation.z);// Set ray hit meta values
		} else {
			optixLaunchParameters.systemState[StartFromScreenBool] = 1;
			// Distribute energy here
		}

	} else { // Visibility check operation
		const unsigned int visibleIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
		optixLaunchParameters.frame.visibleLocations[visibleIndex] = hitLocation;

		colorVector &perRayData = *(colorVector *) getPerRayData<colorVector>();
		perRayData = {sbtData.color.r, sbtData.color.g, sbtData.color.b};
	}
}
extern "C" __global__ void __anyhit__radiance() {}

#pragma clang diagnostic pop