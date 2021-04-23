//
// Created by microbobu on 2/21/21.
//

#ifndef ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
#define ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H

//#include "optix.h"
#include "optix_stubs.h"
#include "types.h"

struct TriangleMeshSBTData {
	float3 *vertex{};
	int3 *index{}; // triangle vertices indices
	colorVector color;
	objectKind kind{};
};

struct RayHitMeta {
	float3 hitLocation;
	float3 from; // Where did this hit originate from
	float3 hitNormal;
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

	unsigned long rayHitIndex = 0; // 1-indexed. 0 = start from screen
	RayHitMeta *rayHitMetas{};
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
