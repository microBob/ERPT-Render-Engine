//
// Created by microbobu on 2/21/21.
//

#ifndef ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
#define ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H

struct OptixLaunchParameters {
	// vec2i fbSize;
	// uint32_t *colorBuffer
	int fbSize;
	int *fbPixels;
};

#endif //ERPT_RENDER_ENGINE_OPTIXLAUNCHPARAMETERS_H
