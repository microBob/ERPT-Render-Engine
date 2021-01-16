//
// Created by microbobu on 1/10/21.
//
#include "../include/transformations.cuh"


float *Transformations::get_worldToCameraMatrix() {
	return worldToCameraMatrix;
}

void Transformations::set_worldToCameraMatrix(float x, float y, float z, float degX, float degY, float degZ) {
	// Convert degrees to radians and negate
	float radX = -degX * (float) M_PI / 180.0f;
	float radY = -degY * (float) M_PI / 180.0f;
	float radZ = -degZ * (float) M_PI / 180.0f;

	/// Copy in matrix
	// 1
	worldToCameraMatrix[0] = cos(radY) * cos(radZ);
	worldToCameraMatrix[1] = sin(radX) * sin(radY) * cos(radZ) + cos(radX) * sin(radZ);
	worldToCameraMatrix[2] = sin(radX) * sin(radZ) - cos(radX) * sin(radY) * cos(radZ);
	// 2
	worldToCameraMatrix[4] = -cos(radY) * sin(radZ);
	worldToCameraMatrix[5] = cos(radX) * cos(radZ) - sin(radX) * sin(radY) * sin(radZ);
	worldToCameraMatrix[6] = cos(radX) * sin(radY) * sin(radZ) + sin(radX) * cos(radZ);
	// 3
	worldToCameraMatrix[8] = sin(radY);
	worldToCameraMatrix[9] = -sin(radX) * cos(radY);
	worldToCameraMatrix[10] = cos(radX) * cos(radY);
	// 4
	worldToCameraMatrix[12] = y * cos(radY) * sin(radZ) - x * cos(radY) * cos(radZ) - z * sin(radY);
	worldToCameraMatrix[13] =
		z * sin(radX) * cos(radY) - x * (sin(radX) * sin(radY) * cos(radZ) + cos(radX) * sin(radZ)) -
		y * (cos(radX) * cos(radZ) - sin(radX) * sin(radY) * sin(radZ));
	worldToCameraMatrix[14] = -x * (sin(radX) * sin(radZ) - cos(radX) * sin(radY) * cos(radZ)) -
	                          y * (cos(radX) * sin(radY) * sin(radZ) + sin(radX) * cos(radZ)) -
	                          z * cos(radX) * cos(radY);
	worldToCameraMatrix[15] = 1.0f;
}

void Transformations::set_perspectiveMatrix(float screenWidth, float screenHeight, float fovRadians, float zFar,
                                            float zNear) {
	perspectiveMatrix[0] = screenWidth / screenHeight / tan(fovRadians / 2);
	perspectiveMatrix[5] = 1.0f / tan(fovRadians / 2);
	perspectiveMatrix[10] = 2.0f / (zFar - zNear);
	perspectiveMatrix[11] = -1.0f;
	perspectiveMatrix[14] = -(zNear - zFar) / (zFar + zNear);
}

float *Transformations::get_perspectiveMatrix() {
	return perspectiveMatrix;
}

void Transformations::convertVerticesToCameraSpace(float *vertices, const int vertexCount) {
	/// Expand worldToCameraMatrix
	// Define and malloc expanded matrix
	float *expandedWorldToCameraMatrix;
	size_t expandedMatrixByteSize = vertexCount * matrixByteSize;
	cudaMallocManaged(&expandedWorldToCameraMatrix, expandedMatrixByteSize);
	cudaMemPrefetchAsync(expandedWorldToCameraMatrix, expandedMatrixByteSize, k.get_cpuID());
	// Copy
	for (int i = 0; i < vertexCount; i += 16) {
		copy(worldToCameraMatrix, worldToCameraMatrix + 16, expandedWorldToCameraMatrix + i);
	}
	// Switch to GPU
	cudaMemAdvise(expandedWorldToCameraMatrix, vertexCount * expandedMatrixByteSize, cudaMemAdviseSetPreferredLocation,
	              k.get_gpuID());
	cudaMemAdvise(expandedWorldToCameraMatrix, vertexCount * expandedMatrixByteSize, cudaMemAdviseSetReadMostly,
	              k.get_gpuID());
	cudaMemPrefetchAsync(expandedWorldToCameraMatrix, vertexCount * expandedMatrixByteSize, k.get_gpuID());

	/// Initialize convertedVertices
	cudaMallocManaged(&convertedVertices, vertexCount * sizeof(float));
	cudaMemAdvise(convertedVertices, vertexCount * sizeof(float), cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemPrefetchAsync(convertedVertices, vertexCount * sizeof(float), k.get_gpuID());

	/// cuBLAS
	status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 1, 4, &alpha, expandedWorldToCameraMatrix,
	                                   4, 16, vertices, 4, 4, &beta, convertedVertices, 4, 4, vertexCount);
	cudaDeviceSynchronize();
	assert(status == CUBLAS_STATUS_SUCCESS);
}

void Transformations::convertToScreenSpace(const int vertexCount) {

}
