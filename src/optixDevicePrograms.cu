#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"
//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

// Launch Parameters
extern "C" __constant__ OptixLaunchParameters optixLaunchParameters;

enum {
	SURFACE_RAY_TYPE = 0,
	RAY_TYPE_COUNT
};

// Payload management
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

// Ray generation program
extern "C" __global__ void __raygen__renderFrame() {
	// Get index and camera
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;
	const auto &camera = optixLaunchParameters.camera;

	// Create per ray data pointer
	colorVector pixelColorPerRayData = {0, 0, 0};
	uint32_t payload0, payload1;
	packPointer(&pixelColorPerRayData, payload0, payload1);

	// Generate ray direction from screen
	const auto screen = make_float2(
		(static_cast<float>(ix) + 0.5f) / static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
		(static_cast<float>(iy) + 0.5f) / static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y));
	auto screenMinus = make_float2(screen.x - 0.5f, screen.y - 0.5f);
	auto horizontalTimesScreenMinus = make_float3(screenMinus.x * camera.horizontal.x,
	                                              screenMinus.x * camera.horizontal.y,
	                                              screenMinus.x * camera.horizontal.z);
	auto verticalTimesScreenMinus = make_float3(screenMinus.y * camera.vertical.x, screenMinus.y * camera.vertical.y,
	                                            screenMinus.y * camera.vertical.z);
	auto rawRayDirection = make_float3(camera.direction.x + horizontalTimesScreenMinus.x + verticalTimesScreenMinus.x,
	                                   camera.direction.y + horizontalTimesScreenMinus.y + verticalTimesScreenMinus.y,
	                                   camera.direction.z + horizontalTimesScreenMinus.z + verticalTimesScreenMinus.z);
	// TODO: can be faster with inverse square root https://www.youtube.com/watch?v=p8u_k2LIZyo
	float rawRayMagnitude = sqrt(pow(rawRayDirection.x, 2) +
	                             pow(rawRayDirection.y, 2) +
	                             pow(rawRayDirection.z, 2));
	auto rayDirectionNormalized = make_float3(rawRayDirection.x / rawRayMagnitude,
	                                          rawRayDirection.y / rawRayMagnitude,
	                                          rawRayDirection.z / rawRayMagnitude);

	if (ix == static_cast<unsigned int>(optixGetLaunchDimensions().x / 2) &&
	    iy == static_cast<unsigned int>(optixGetLaunchDimensions().y / 2)) {
		printf("ix, iy:\t%u, %u\n", ix, iy);
		printf("screen:\t%f, %f\n", screen.x, screen.y);
		printf("screenMinus:\t%f, %f\n", screenMinus.x, screenMinus.y);
		printf("horizontalTimesScreenMinus:\t%f, %f, %f\n", horizontalTimesScreenMinus.x, horizontalTimesScreenMinus.y,
		       horizontalTimesScreenMinus.z);
		printf("verticalTimesScreenMinus:\t%f, %f, %f\n", verticalTimesScreenMinus.x, verticalTimesScreenMinus.y, verticalTimesScreenMinus.z);
		printf("rawRayDirection:\t%f, %f, %f\n", rawRayDirection.x, rawRayDirection.y, rawRayDirection.z);
		printf("rawRayMagnitude:\t%f\n", rawRayMagnitude);
		printf("rayDirectionNormalized:\t%f, %f, %f\n", rayDirectionNormalized.x, rayDirectionNormalized.y, rayDirectionNormalized.z);
	}

	// Optix Trace
	optixTrace(optixLaunchParameters.optixTraversableHandle,
	           camera.position,
	           rayDirectionNormalized,
	           0.f,
	           1e20f,
	           0.0f,
	           OptixVisibilityMask(255),
	           OPTIX_RAY_FLAG_DISABLE_ANYHIT,
	           SURFACE_RAY_TYPE,
	           RAY_TYPE_COUNT,
	           SURFACE_RAY_TYPE,
	           payload0, payload1);

	const unsigned int colorBufferIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	optixLaunchParameters.frame.frameColorBuffer[colorBufferIndex] = pixelColorPerRayData;
}

// Miss program
extern "C" __global__ void __miss__radiance() {
	colorVector &perRayData = *(colorVector *) getPerRayData<colorVector>();
	perRayData = {1, 1, 1}; // Set to white
}

// Hit program
extern "C" __global__ void __closesthit__radiance() {
	colorVector &perRayData = *(colorVector *) getPerRayData<colorVector>();
	perRayData = {0, 0, 0};
}
extern "C" __global__ void __anyhit__radiance() {}

#pragma clang diagnostic pop