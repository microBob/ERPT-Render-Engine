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

	// Single matrix instance (expand for cublas by implementation)
	float *worldToPerspectiveMatrix = (float *) malloc(matrixByteSize);

	// Converted vertices
	float *perspectiveVertices;
	float *screenCoordinates;
public:
	Transformations() {
		cublasCreate(&handle);
	}

	void set_worldToPerspectiveMatrix(float x, float y, float z, float degX, float degY, float degZ, float fov,
	                                  float screenWidth, float screenHeight, float zNear, float zFar);

	void convertWorldToPerspectiveSpace(float *vertices, const int vertexCount);

	void convertPerspectiveToScreenSpace(const int vertexCount, float screenWidth, float screenHeight);

	void cleanup() {
		free(worldToPerspectiveMatrix);

		cudaFree(perspectiveVertices);

		cublasDestroy(handle);
	}
};

#endif //ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
