//
// Created by microbobu on 1/10/21.
//
#include "../include/kernels.cuh"

Kernels::Kernels() { // NOLINT(cppcoreguidelines-pro-type-member-init)
	cudaGetDeviceProperties(&prop, gpuID);
}

void Kernels::set_kernelThreadsAndBlocks(int sceneVertexCount) {
	threadsToLaunchForVertices = min(prop.maxThreadsPerBlock, (int) sceneVertexCount);
	blocksToLaunchForVertices = min(prop.maxGridSize[0],
	                                1 + (sceneVertexCount / prop.maxThreadsPerBlock));
}
