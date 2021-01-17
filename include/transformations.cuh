//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
#define ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH

//// SECTION: Include
#include "kernels.cuh"
#include "cublas_v2.h"

#include <iostream>
#include <cassert>


//// SECTION: Class definition
class Transformations {
private:
	// cuBLAS
	cublasHandle_t handle;
	cublasStatus_t status;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	// Matrix sizes
	const size_t matrixByteSize = 16 * sizeof(float);
	size_t expandedMatrixByteSize;
	size_t vertexCoCount;

	// Single matrix instance (expand for cublas by implementation)
	float *worldToCameraMatrix = (float *) malloc(matrixByteSize);
	float *perspectiveMatrix = (float *) malloc(matrixByteSize);

	// Converted vertices
	float *cameraVertices;
	float *perspectiveVertices;
	float *screenCoordinates;
public:
	Transformations() {
		cublasCreate(&handle);
	}

	float *get_worldToCameraMatrix();

	void set_worldToCameraMatrix(float x, float y, float z, float degX, float degY, float degZ);

	void set_perspectiveMatrix(float screenWidth, float screenHeight, float fovRadians, float zFar, float zNear);

	float *get_perspectiveMatrix();

	void convertVerticesToCameraSpace(float *vertices, const int vertexCount);

	void convertToPerspectiveSpace(const int vertexCount);

	void convertToScreenSpace(const int vertexCount, float screenWidth, float screenHeight);

	void cleanup() {
		free(worldToCameraMatrix);
		free(perspectiveMatrix);

		cudaFree(cameraVertices);
		cudaFree(perspectiveVertices);

		cublasDestroy(handle);
	}
};

#endif //ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
