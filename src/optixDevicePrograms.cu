//
// Created by microbobu on 2/21/21.
//
#include "../include/optixLaunchParameters.h"
#include "optix_device.h"

// Launch Parameters
extern "C" __constant__ OptixLaunchParameters launchParameters;

// Ray generation program
extern "C" __global__ void __raygen__renderFrame() {

}

// Miss program
extern "C" __global__ void __miss__radiance() {}

// Hit program
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
