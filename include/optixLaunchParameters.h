//
// Created by microbobu on 2/21/21.
//

#ifndef ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
#define ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H

//#include "optix.h"
#include "optix_stubs.h"
#include "types.h"

struct TriangleMeshSBTData {
	colorVector color;
	float3 *vertex{};
	int3 *index{}; // triangle vertices indices
};

struct RayHitMeta {
	float3 location;
	float3 from; // Where did this hit originate from
	float rayLength;
	bool cameraVisible;
	unsigned long visits; // For detailed balance
	unsigned long raysFromThisPoint; // Subsequent rays from this point
	int energy; // Brightness
};

struct OptixLaunchParameters {
	struct {
		colorVector *frameColorBuffer{};
		uint2 frameBufferSize{};
	} frame;

	struct {
		float3 position, direction, horizontal, vertical;
	} camera{};

	OptixTraversableHandle optixTraversableHandle{};

	unsigned long mutationNumbersIndex = 0;
	float3 *mutationNumbers{};

	RayHitMeta *rayHitMetas{};
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
