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
	MeshKind kind{};
};

struct RayHitMeta {
	float3 hitLocation;
	float3 from; // Where did this hit originate from
	float3 hitNormal;
	float rayLength;
	unsigned long visits; // For detailed balance
	bool isRootRay;
	unsigned long sourceRayIndex; // Subsequent rays from this point
	float energy; // Brightness
};

struct OptixLaunchParameters {
	struct {
		colorVector *frameColorBuffer{};
		uint2 frameBufferSize{};
		float3 *visibleLocations{};
	} frame;

	struct {
		float3 position, direction, horizontal, vertical;
	} camera{};

	OptixTraversableHandle optixTraversableHandle{};

	struct {
		size_t numberOfThem{};
		float3 *numbers{};
	} mutation;

	struct {
		float visibilityTolerance = 0.01;
		RayHitMeta *metas{};
	} rayHit;

	unsigned long *systemState{};
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
