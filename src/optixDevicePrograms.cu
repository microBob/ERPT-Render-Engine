//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

// Launch Parameters
extern "C" __constant__ OptixLaunchParameters optixLaunchParameters;

// Ray generation program
extern "C" __global__ void __raygen__renderFrame() {
	// Sanity check
	if (optixLaunchParameters.frameID == 0 && optixGetLaunchIndex().x == 0 && optixGetLaunchIndex().y == 0) {
		printf("Hello world from OptiX!\n");
		printf("Launch Size: %i x %i\n", optixLaunchParameters.frameBufferSize.x,
		       optixLaunchParameters.frameBufferSize.y);
	}

	// Create test pattern
	const unsigned int ix = optixGetLaunchIndex().x;
	const unsigned int iy = optixGetLaunchIndex().y;

	const float r = static_cast<float>(ix % 256) / 255.0f;
	const float g = static_cast<float>(iy % 256) / 255.0f;
	const float b = static_cast<float>((ix + iy) % 256) / 255.0f;

	const colorVector pixelColor = {r, g, b, 1.0};
	const unsigned int colorBufferIndex = ix+iy*optixLaunchParameters.frameBufferSize.x;
	optixLaunchParameters.frameColorBuffer[colorBufferIndex] = pixelColor;
}

// Miss program
extern "C" __global__ void __miss__radiance() {}

// Hit program
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
