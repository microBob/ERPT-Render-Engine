//
// Created by microbobu on 1/10/21.
//
#include "../include/transformations.cuh"


Transformations::Transformations() {
	// Initialize into memory
	cudaMallocManaged(&worldToCameraMatrix, matrixByteSize);
	cudaMallocManaged(&perspectiveMatrix, matrixByteSize);
}

float *Transformations::get_worldToCameraMatrix() {
	return worldToCameraMatrix;
}

void Transformations::set_worldToCameraMatrix(float x, float y, float z, float degX, float degY, float degZ) {
	cudaMemAdvise(worldToCameraMatrix, matrixByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());

	// Convert degrees to radians and negate
	float radX = -degX * (float) M_PI / 180.0f;
	float radY = -degY * (float) M_PI / 180.0f;
	float radZ = -degZ * (float) M_PI / 180.0f;

	worldToCameraMatrix[0] = cos(radY) * cos(radZ);
	worldToCameraMatrix[1] = sin(radX) * sin(radY) * cos(radZ) + cos(radX) * sin(radZ);
	worldToCameraMatrix[2] = sin(radX) * sin(radZ) - cos(radX) * sin(radY) * cos(radZ);

	worldToCameraMatrix[4] = -cos(radY) * sin(radZ);
	worldToCameraMatrix[5] = cos(radX) * cos(radZ) - sin(radX) * sin(radY) * sin(radZ);
	worldToCameraMatrix[6] = cos(radX) * sin(radY) * sin(radZ) + sin(radX) * cos(radZ);

	worldToCameraMatrix[8] = sin(radY);
	worldToCameraMatrix[9] = -sin(radX) * cos(radY);
	worldToCameraMatrix[10] = cos(radX) * cos(radY);

	worldToCameraMatrix[12] = y * cos(radY) * sin(radZ) - x * cos(radY) * cos(radZ) - z * sin(radY);
	worldToCameraMatrix[13] =
		z * sin(radX) * cos(radY) - x * (sin(radX) * sin(radY) * cos(radZ) + cos(radX) * sin(radZ)) -
		y * (cos(radX) * cos(radZ) - sin(radX) * sin(radY) * sin(radZ));
	worldToCameraMatrix[14] = -x * (sin(radX) * sin(radZ) - cos(radX) * sin(radY) * cos(radZ)) -
	                          y * (cos(radX) * sin(radY) * sin(radZ) + sin(radX) * cos(radZ)) -
	                          z * cos(radX) * cos(radY);
	worldToCameraMatrix[15] = 1.0f;

	// Prepare for reading
	cudaMemAdvise(worldToCameraMatrix, matrixByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemAdvise(worldToCameraMatrix, matrixByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(worldToCameraMatrix, matrixByteSize, k.get_gpuID());
}

void Transformations::set_perspectiveMatrix(float screenWidth, float screenHeight, float fovRadians, float zFar,
                                            float zNear) {
	cudaMemAdvise(perspectiveMatrix, matrixByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());

	perspectiveMatrix[0] = screenWidth / screenHeight / tan(fovRadians / 2);
	perspectiveMatrix[5] = 1.0f / tan(fovRadians / 2);
	perspectiveMatrix[10] = 2.0f / (zFar - zNear);
	perspectiveMatrix[11] = -1.0f;
	perspectiveMatrix[14] = -(zNear - zFar) / (zFar + zNear);

	cudaMemAdvise(perspectiveMatrix, matrixByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemAdvise(perspectiveMatrix, matrixByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(perspectiveMatrix, matrixByteSize, k.get_gpuID());
}

float *Transformations::get_perspectiveMatrix() {
	return perspectiveMatrix;
}