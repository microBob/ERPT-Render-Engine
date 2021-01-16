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
	Kernels k;
	static const size_t matrixByteSize = 16 * sizeof(float);

	float *worldToCameraMatrix{};
	float *perspectiveMatrix{};

	float *convertedVertices;
public:
	Transformations();

	float *get_worldToCameraMatrix();

	void set_worldToCameraMatrix(float x, float y, float z, float degX, float degY, float degZ);

	void set_perspectiveMatrix(float screenWidth, float screenHeight, float fovRadians, float zFar, float zNear);

	float *get_perspectiveMatrix();

	void convertVerticesToCameraSpace(float *vertices);

};

#endif //ERPT_RENDER_ENGINE_TRANSFORMATIONS_CUH
