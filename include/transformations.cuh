//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
#define ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH

//// SECTION: Include
#include "kernels.cuh"
#include "cublas_v2.h"


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

	// Single matrix instance (expand for cublas by implementation)
	float *worldToPerspectiveMatrix = (float *) malloc(matrixByteSize);

public:
	Transformations() {
		cublasCreate(&handle);
	}

	// Perspective projection
	void
	set_worldToPerspectiveMatrix(float locX, float locY, float locZ, float rotX, float rotY, float rotZ,
	                             float fov, float screenRatio, float zNear, float zFar);

	void convertWorldToPerspectiveSpace(float *input, const int vertexCount, float *output);

	void convertPerspectiveToScreenSpace(float *input, const int vertexCount, float screenWidth,
	                                     float screenHeight, unsigned int blocks, unsigned int threads, float *output);

	static void convertPerspectiveToScreenSpaceCPU(float screenWidth, float screenHeight, int sceneVertexCount,
	                                               const float *perspectiveVertices, float *screenCoordinates);

	// Cleanup
	void cleanup() {
		free(worldToPerspectiveMatrix);

		cublasDestroy(handle);
	}
};

#endif //ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
