//
// Created by microbobu on 2/15/21.
//

#ifndef ERPT_RENDER_ENGINE_DRAWINGS_CUH
#define ERPT_RENDER_ENGINE_DRAWINGS_CUH


//// SECTION: Include
#include "kernels.cuh"

#include <iostream>
#include <vector>
#include "constants.h"
// RapidJSON
#include "rapidjson/document.h"


//// SECTION: Class definition
class Drawings {
public:
	static unsigned int cartesianToLinear(float x, float y, float screenWidth);

	static void drawDotCPU(float x, float y, float *output, float screenWidth);

	static vector<vector<unsigned int>> extractConnectedVerticesCPU(const rapidjson::GenericValue<rapidjson::UTF8<>>::Array &meshDataDOM);

	static void drawWireframeCPU(float screenWidth, float screenHeight, float *pixData, const float *screenCoordinates,
	                      vector<vector<unsigned int>> &connectedVertices);
};

#endif //ERPT_RENDER_ENGINE_DRAWINGS_CUH{};