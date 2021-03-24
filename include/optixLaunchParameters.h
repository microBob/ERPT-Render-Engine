//
// Created by microbobu on 2/21/21.
//

#ifndef ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
#define ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H

#include "optix.h"
#include "optix_stubs.h"
#include "types.h"

struct OptixLaunchParameters {
	struct {
		colorVector *frameColorBuffer{};
		vector2i frameBufferSize{};
	} frame;

	struct {
		vector3 position, direction, horizontal, vertical;
	} camera{};

	OptixTraversableHandle optixTraversableHandle{};
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
