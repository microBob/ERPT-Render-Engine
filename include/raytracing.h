//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_RAYTRACING_H
#define ERPT_RENDER_ENGINE_RAYTRACING_H

//// SECTION: Includes
#include "../include/kernels.cuh"
#include "optix.h"
#include "optix_stubs.h"

//// SECTION: Class definition
class Raytracing{
public:
	static void initOptix();
};
#endif //ERPT_RENDER_ENGINE_RAYTRACING_H