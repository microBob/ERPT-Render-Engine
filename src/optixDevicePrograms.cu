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
	// Create test pattern
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;

	const float r = static_cast<float>(ix % 256) / 255.0f;
	const float g = static_cast<float>(iy % 256) / 255.0f;
	const float b = static_cast<float>((ix + iy) % 256) / 255.0f;

	const colorVector pixelColor = {r, g, b};
	const unsigned int colorBufferIndex = ix + iy * optixLaunchParameters.frame.frameBufferSize.x;
	optixLaunchParameters.frame.frameColorBuffer[colorBufferIndex] = pixelColor;
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