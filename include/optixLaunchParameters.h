//
// Created by microbobu on 2/21/21.
//

#ifndef ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
#define ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H

//#include "optix.h"
#include "optix_stubs.h"
#include "curand_kernel.h"
#include "types.h"

struct TriangleMeshSBTData {
	float3 *vertex{};
	int3 *index{}; // triangle vertices indices
	colorVector color;
	float energy = 1;
	MeshKind kind{};
};

struct PerRayData {
	float3 location{}; // Where the trace hit
	float3 normal{}; // Trace hit normal
	float3 xAxis{};
	float3 yAxis{};
	colorVector color;
	float energy = 1;
	bool light{}; // Was it a light source
};

struct OptixLaunchParameters {
	struct {
		colorVector *frameColorBuffer{};
		uint2 frameBufferSize{};
	} frame;

	struct {
		float3 position, direction, horizontal, vertical;
	} camera{};

	struct {
		unsigned long index;
		unsigned long total;
	} samples{};

	unsigned int traceDepth{};

	OptixTraversableHandle optixTraversableHandle{};

	float *curMutationNumbers{};
	float *newMutationNumbers{};

	float *energyPerPixel{}; // Energy level per pixel
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
