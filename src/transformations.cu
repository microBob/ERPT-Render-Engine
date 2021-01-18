//
// Created by microbobu on 1/10/21.
//
#include "../include/transformations.cuh"

//// SECTION: Manual kernels and functions
__device__ unsigned int sceneToLinearGPU(unsigned int vertex, int coordinate, int dim) {
	return vertex * dim + coordinate;
}

__global__ void
convertToScreenSpaceKernel(float *input, const int vertexCount, float screenWidth, float screenHeight, float *output) {
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int increment = blockDim.x * gridDim.x;

	while (tid < vertexCount) {
		// Skip if point is outside of z-cull space
		if (abs(input[sceneToLinearGPU(tid, 2, 0)]) > 1) {
			tid += increment;
			continue;
		}
		// Skip if point will cause divide by 0
		if (input[sceneToLinearGPU(tid, 3, 0)] == 0) {
			tid += increment;
			continue;
		}

		// Calculate final x and y
		output[sceneToLinearGPU(tid, 0, 2)] =
			input[sceneToLinearGPU(tid, 0, 4)] / input[sceneToLinearGPU(tid, 3, 4)] * screenWidth;
		output[sceneToLinearGPU(tid, 1, 2)] =
			input[sceneToLinearGPU(tid, 1, 4)] / input[sceneToLinearGPU(tid, 3, 4)] * screenHeight;

		// Increment tid
		tid += increment;
	}
}


//// SECTION: Transformations implementation
void
Transformations::set_worldToPerspectiveMatrix(float x, float y, float z, float degX, float degY, float degZ, float fov,
                                              float screenWidth, float screenHeight, float zNear, float zFar) {
	// Convert degrees to radians (negate already built into matrix)
	float radX = degX * (float) M_PI / 180.0f;
	float radY = degY * (float) M_PI / 180.0f;
	float radZ = degZ * (float) M_PI / 180.0f;

	// Common values
	float tanFov = tan(fov / 2);
	float wTanFov = screenWidth * tanFov;
	float nearFar = zNear - zFar;
	float comExpr1 = (-cos(radX) * sin(radZ) + cos(radZ) * sin(radX) * sin(radY)) / tanFov;
	float comExpr2 = (-2 * sin(radX) * sin(radZ) - 2 * cos(radX) * cos(radZ) * sin(radY)) / nearFar;
	float comExpr3 = -sin(radX) * sin(radZ) - cos(radX) * cos(radZ) * sin(radY);
	float comExpr4 = (cos(radX) * cos(radZ) + sin(radX) * sin(radY) * sin(radZ)) / tanFov;
	float comExpr5 = (2 * cos(radZ) * sin(radX) - 2 * cos(radX) * sin(radY) * sin(radZ)) / nearFar;
	float comExpr6 = cos(radZ) * sin(radX) - cos(radX) * sin(radY) * sin(radZ);
	float comExpr7 = cos(radY) * sin(radX) / tanFov;

	/// Copy in matrix
	// 1
	worldToPerspectiveMatrix[0] = screenHeight * cos(radY) * cos(radZ) / wTanFov;
	worldToPerspectiveMatrix[1] = comExpr1;
	worldToPerspectiveMatrix[2] = comExpr2;
	worldToPerspectiveMatrix[3] = comExpr3;
	// 2
	worldToPerspectiveMatrix[4] = -screenHeight * cos(radY) * sin(radZ) / wTanFov;
	worldToPerspectiveMatrix[5] = comExpr4;
	worldToPerspectiveMatrix[6] = comExpr5;
	worldToPerspectiveMatrix[7] = comExpr6;
	// 3
	worldToPerspectiveMatrix[8] = -screenHeight * sin(radY) / wTanFov;
	worldToPerspectiveMatrix[9] = comExpr7;
	worldToPerspectiveMatrix[10] = -2 * cos(radX) * cos(radY) / nearFar;
	worldToPerspectiveMatrix[11] = -cos(radX) * cos(radY);
	// 4
	worldToPerspectiveMatrix[12] =
		(-cos(radZ) * screenHeight * x * cos(radY) - sin(radZ) * screenHeight * y * cos(radY) +
		 screenHeight * z * sin(radY)) / wTanFov;
	worldToPerspectiveMatrix[13] = -x * comExpr1 - y * comExpr2 - z * comExpr7;
	worldToPerspectiveMatrix[14] =
		-x * comExpr2 - y * comExpr5 + (-zNear + zFar) / (zNear + zFar) + 2 * cos(radX) * cos(radY) / -nearFar;
	worldToPerspectiveMatrix[15] = -x * comExpr3 - y * comExpr6 + z * cos(radX) * cos(radY);
}

void Transformations::convertWorldToPerspectiveSpace(float *input, const int vertexCount, float *output) {
	/// Expand worldToCameraMatrix
	// Define and malloc expanded matrix
	float *expandedWorldToPerspectiveMatrix;
	size_t expandedMatrixByteSize = vertexCount * matrixByteSize;
	cudaMallocManaged(&expandedWorldToPerspectiveMatrix, expandedMatrixByteSize);
	cudaMemPrefetchAsync(expandedWorldToPerspectiveMatrix, expandedMatrixByteSize, k.get_cpuID());
	// Copy (expand)
	for (int i = 0; i < vertexCount; ++i) {
		copy(worldToPerspectiveMatrix, worldToPerspectiveMatrix + 16, expandedWorldToPerspectiveMatrix + i * 16);
	}
	// Switch to GPU
	cudaMemAdvise(expandedWorldToPerspectiveMatrix, expandedMatrixByteSize,
	              cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemAdvise(expandedWorldToPerspectiveMatrix, expandedMatrixByteSize, cudaMemAdviseSetReadMostly,
	              k.get_gpuID());
	cudaMemPrefetchAsync(expandedWorldToPerspectiveMatrix, expandedMatrixByteSize, k.get_gpuID());

	/// cuBLAS
	status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 1, 4, &alpha,
	                                   expandedWorldToPerspectiveMatrix,
	                                   4, 16, input, 4, 4, &beta, output, 4, 4, vertexCount);
	cudaDeviceSynchronize();
	assert(status == CUBLAS_STATUS_SUCCESS);

	// Cleanup
	cudaFree(expandedWorldToPerspectiveMatrix);
}

void Transformations::convertPerspectiveToScreenSpace(float *input, const int vertexCount, float screenWidth,
                                                      float screenHeight,
                                                      float *output) {
	// Define and malloc screenCoordinates
	cudaMallocManaged(&output, 2 * vertexCount * sizeof(float));
	cudaMemPrefetchAsync(output, 2 * vertexCount * sizeof(float), k.get_gpuID());

	// Half screen dimension
	float halfWidth = screenWidth / 2;
	float halfHeight = screenHeight / 2;

	// Run kernel
	convertToScreenSpaceKernel<<<k.get_blocksToLaunchForVertices(), k.get_threadsToLaunchForVertices()>>>(
		input, vertexCount, halfWidth, halfHeight, output);
	cudaDeviceSynchronize();
}