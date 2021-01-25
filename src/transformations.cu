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
                                              float fov, float screenRatio, float zNear, float zFar) {
	// Common values
	float tanFov = tan(fov / 2);
	float zClip = zFar - zNear;
	float comExp1 = sin(rotX) * sin(rotZ);
	float comExp2 = sin(rotY) * cos(rotX) * cos(rotZ);
	float comExp3 = sin(rotY) * sin(rotZ) * cos(rotX);

	// Set Matrix
	worldToPerspectiveMatrix[0] = cos(rotY) * cos(rotZ) / tanFov;
	worldToPerspectiveMatrix[1] = screenRatio * (sin(rotX) * sin(rotY) * cos(rotZ) - sin(rotZ) * cos(rotX)) / tanFov;
	worldToPerspectiveMatrix[2] = 2 * (comExp1 + comExp2) / zClip;
	worldToPerspectiveMatrix[3] = -sin(rotX) * sin(rotZ) - comExp2;

	worldToPerspectiveMatrix[4] = sin(rotZ) * cos(rotY) / tanFov;
	worldToPerspectiveMatrix[5] = screenRatio * (sin(rotX) * sin(rotY) * sin(rotZ) + cos(rotX) * cos(rotZ)) / tanFov;
	worldToPerspectiveMatrix[6] = 2 * (-sin(rotX) * cos(rotZ) + comExp3) / zClip;
	worldToPerspectiveMatrix[7] = sin(rotX) * cos(rotZ) - comExp3;

	worldToPerspectiveMatrix[8] = -sin(rotY) / tanFov;
	worldToPerspectiveMatrix[9] = screenRatio * sin(rotX) * cos(rotY) / tanFov;
	worldToPerspectiveMatrix[10] = 2 * cos(rotX) * cos(rotY) / zClip;
	worldToPerspectiveMatrix[11] = -cos(rotX) * cos(rotY);

	worldToPerspectiveMatrix[12] =
		(-locX * cos(rotY) * cos(rotZ) - locY * sin(rotZ) * cos(rotY) + locZ * sin(rotY)) / tanFov;
	worldToPerspectiveMatrix[13] = -screenRatio * (locX * (sin(rotX) * sin(rotY) * cos(rotZ) - sin(rotZ) * cos(rotX)) +
	                                               locY * (sin(rotX) * sin(rotY) * sin(rotZ) + cos(rotX) * cos(rotZ)) +
	                                               locZ * sin(rotX) * cos(rotY)) /
	                               tanFov;
	worldToPerspectiveMatrix[14] = ((float) pow(zClip, 2) + 2 * (zFar + zNear) * (-locX * (comExp1 +
	                                                                                       comExp2) +
	                                                                              locY * (sin(rotX) * cos(rotZ) -
	                                                                                      comExp3) -
	                                                                              locZ * cos(rotX) * cos(rotY))) / (
		                               zClip * (zFar + zNear));
	worldToPerspectiveMatrix[15] = locX * (comExp1 + comExp2) - locY * (sin(rotX) * cos(rotZ) -
	                                                                    comExp3) + locZ * cos(rotX) * cos(rotY);
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
                                                      float screenHeight, unsigned int blocks, unsigned int threads,
                                                      float *output) {
	// Half screen dimension
	float halfWidth = screenWidth / 2;
	float halfHeight = screenHeight / 2;

	cout << "blocks: " << blocks << endl;
	cout << "threads: " << threads << endl << endl;

	// Run kernel
	convertToScreenSpaceKernel<<<blocks, threads>>>(
		input, vertexCount, halfWidth, halfHeight, output);
	cudaDeviceSynchronize();
//
//	for (int i = 0; i < vertexCount * 3; ++i) {
//		cout << output[i];
//		if ((i + 1) % 3 == 0) {
//			cout << endl;
//		} else {
//			cout << ",\t";
//		}
//	}
//	cout << endl;
}