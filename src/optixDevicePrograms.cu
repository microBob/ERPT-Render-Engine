#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"
//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

// Launch Parameters
extern "C" __constant__ OptixLaunchParameters optixLaunchParameters;

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
	colorVector pixelColorPerRayData{};
	uint32_t payload0, payload1;
	packPointer(&pixelColorPerRayData, payload0, payload1);

	// Generate ray direction from screen
	const vector2f screen = {
		(static_cast<float>(ix) + 0.5f) / static_cast<float>(optixLaunchParameters.frame.frameBufferSize.x),
		(static_cast<float>(iy) + 0.5f) / static_cast<float>(optixLaunchParameters.frame.frameBufferSize.y)
	};
	vector2f screenMinus = {screen.x - 0.5f, screen.y - 0.5f};
	vector3 rawRayDirection = {
		camera.direction.x + screenMinus.x * camera.horizontal.x + screenMinus.y * camera.vertical.x,
		camera.direction.y + screenMinus.x * camera.horizontal.y + screenMinus.y * camera.vertical.y,
		camera.direction.z + screenMinus.x * camera.horizontal.z + screenMinus.y * camera.vertical.z
	};
	float rawRayMagnitude = sqrt(pow(rawRayDirection.x, 2) +
	                             pow(rawRayDirection.y, 2) +
	                             pow(rawRayDirection.z, 2));
	vector3 rayDirectionNormalized = {
		rawRayDirection.x / rawRayMagnitude,
		rawRayDirection.y / rawRayMagnitude,
		rawRayDirection.z / rawRayMagnitude
	};

	// Optix Trace
	optixTrace(optixLaunchParameters.optixTraversableHandle,
	           camera.position,
	           rayDirectionNormalized,
	           0.f,    // tmin
	           1e20f,  // tmax
	           0.0f,   // rayTime
	           OptixVisibilityMask(255),
	           OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
	           0,             // SBT offset
	           1,               // SBT stride
	           0,             // missSBTIndex
	           payload0, payload1);

	const unsigned int colorBufferIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	optixLaunchParameters.frame.frameColorBuffer[colorBufferIndex] = pixelColorPerRayData;
}

// Miss program
extern "C" __global__ void __miss__radiance() {
	colorVector &perRayData = *(colorVector *) getPerRayData<colorVector>();
	perRayData = {0, 0, 0}; // Set to black
}

// Hit program
extern "C" __global__ void __closesthit__radiance() {
	const unsigned int primitiveIdx = optixGetPrimitiveIndex();
	colorVector &perRayData = *(colorVector *) getPerRayData<colorVector>();
	float shade = static_cast<float>(primitiveIdx % 256) / 255.0f;
	perRayData = {shade, shade, shade}; // Some shade of grey
}
extern "C" __global__ void __anyhit__radiance() {}

#pragma clang diagnostic pop