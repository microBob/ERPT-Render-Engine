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
		if (abs(input[sceneToLinearGPU(tid, 2, 4)]) > 1) {
			tid += increment;
			continue;
		}
		// Skip if point will cause divide by 0
		if (input[sceneToLinearGPU(tid, 3, 4)] == 0) {
			tid += increment;
			continue;
		}

		// Calculate final x and y
		output[sceneToLinearGPU(tid, 0, 3)] =
			(input[sceneToLinearGPU(tid, 0, 4)] / input[sceneToLinearGPU(tid, 3, 4)] + 1) * screenWidth;
		output[sceneToLinearGPU(tid, 1, 3)] =
			(input[sceneToLinearGPU(tid, 1, 4)] / input[sceneToLinearGPU(tid, 3, 4)] + 1) * screenHeight;
		output[sceneToLinearGPU(tid, 2, 3)] = input[sceneToLinearGPU(tid, 3, 4)];

		// Increment tid
		tid += increment;
	}
}


//// SECTION: Transformations implementation
void
Transformations::set_worldToPerspectiveMatrix(float locX, float locY, float locZ, float rotX, float rotY, float rotZ,
                                              float fov,
                                              float screenWidth, float screenHeight, float zNear, float zFar) {
	// Common values
	float tanFov = tan(fov / 2);
	float wTanFov = screenWidth * tanFov;
	float nearFar = zNear - zFar;
	float comExpr1 = (-cos(rotX) * sin(rotZ) + cos(rotZ) * sin(rotX) * sin(rotY)) / tanFov;
	float comExpr2 = (-2 * sin(rotX) * sin(rotZ) - 2 * cos(rotX) * cos(rotZ) * sin(rotY)) / nearFar;
	float comExpr3 = -sin(rotX) * sin(rotZ) - cos(rotX) * cos(rotZ) * sin(rotY);
	float comExpr4 = (cos(rotX) * cos(rotZ) + sin(rotX) * sin(rotY) * sin(rotZ)) / tanFov;
	float comExpr5 = (2 * cos(rotZ) * sin(rotX) - 2 * cos(rotX) * sin(rotY) * sin(rotZ)) / nearFar;
	float comExpr6 = cos(rotZ) * sin(rotX) - cos(rotX) * sin(rotY) * sin(rotZ);
	float comExpr7 = cos(rotY) * sin(rotX) / tanFov;

	/// Copy in matrix
	// 1
	worldToPerspectiveMatrix[0] = screenHeight * cos(rotY) * cos(rotZ) / wTanFov;
	worldToPerspectiveMatrix[1] = comExpr1;
	worldToPerspectiveMatrix[2] = comExpr2;
	worldToPerspectiveMatrix[3] = comExpr3;
	// 2
	worldToPerspectiveMatrix[4] = screenHeight * cos(rotY) * sin(rotZ) / wTanFov;
	worldToPerspectiveMatrix[5] = comExpr4;
	worldToPerspectiveMatrix[6] = comExpr5;
	worldToPerspectiveMatrix[7] = comExpr6;
	// 3
	worldToPerspectiveMatrix[8] = -screenHeight * sin(rotY) / wTanFov;
	worldToPerspectiveMatrix[9] = comExpr7;
	worldToPerspectiveMatrix[10] = -2 * cos(rotX) * cos(rotY) / nearFar;
	worldToPerspectiveMatrix[11] = -cos(rotX) * cos(rotY);
	// 4
	worldToPerspectiveMatrix[12] =
		(-cos(rotZ) * screenHeight * locX * cos(rotY) - sin(rotZ) * screenHeight * locY * cos(rotY) +
		 screenHeight * locZ * sin(rotY)) / wTanFov;
	worldToPerspectiveMatrix[13] = -(locX * (sin(rotX) * sin(rotY) * cos(rotZ) - sin(rotZ) * cos(rotX)) +
	                                 locY * (sin(rotX) * sin(rotY) * sin(rotZ) + cos(rotX) * cos(rotZ)) +
	                                 locZ * sin(rotX) * cos(rotY)) / tanFov;
	worldToPerspectiveMatrix[14] = ((float) pow(zFar - zNear, 2) + 2 * (zFar + zNear) *
	                                                               (-locX * (sin(rotX) * sin(rotZ) +
	                                                                         sin(rotY) * cos(rotX) * cos(rotZ)) +
	                                                                locY * (sin(rotX) * cos(rotZ) -
	                                                                        sin(rotY) * sin(rotZ) * cos(rotX)) -
	                                                                locZ * cos(rotX) * cos(rotY))) /
	                               ((zFar - zNear) * (zFar + zNear));
	worldToPerspectiveMatrix[15] = -locX * comExpr3 - locY * comExpr6 + locZ * cos(rotX) * cos(rotY);
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
	// Half screen dimension
	float halfWidth = screenWidth / 2;
	float halfHeight = screenHeight / 2;

	// Run kernel
	convertToScreenSpaceKernel<<<k.get_blocksToLaunchForVertices(), k.get_threadsToLaunchForVertices()>>>(
		input, vertexCount, halfWidth, halfHeight, output);
	cudaDeviceSynchronize();
}