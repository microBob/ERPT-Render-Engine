//
// Created by microbobu on 2/15/21.
//
#include "../include/drawings.cuh"

unsigned int Drawings::cartesianToLinear(float x, float y, float screenWidth) {
	return (unsigned int) (round(y) * screenWidth + round(x));
}

void Drawings::drawDotCPU(float x, float y, float *output, float screenWidth) {
	unsigned int screenCoordinate = cartesianToLinear(x, y, screenWidth);
	output[screenCoordinate * 4] = 1.0f;
	output[screenCoordinate * 4 + 1] = 1.0f;
	output[screenCoordinate * 4 + 2] = 1.0f;
	output[screenCoordinate * 4 + 3] = 1.0f;
	screenCoordinate = cartesianToLinear(x + 1, y, screenWidth);
	output[screenCoordinate * 4] = 1.0f;
	output[screenCoordinate * 4 + 1] = 1.0f;
	output[screenCoordinate * 4 + 2] = 1.0f;
	output[screenCoordinate * 4 + 3] = 1.0f;
	screenCoordinate = cartesianToLinear(x, y + 1, screenWidth);
	output[screenCoordinate * 4] = 1.0f;
	output[screenCoordinate * 4 + 1] = 1.0f;
	output[screenCoordinate * 4 + 2] = 1.0f;
	output[screenCoordinate * 4 + 3] = 1.0f;
	screenCoordinate = cartesianToLinear(x - 1, y, screenWidth);
	output[screenCoordinate * 4] = 1.0f;
	output[screenCoordinate * 4 + 1] = 1.0f;
	output[screenCoordinate * 4 + 2] = 1.0f;
	output[screenCoordinate * 4 + 3] = 1.0f;
	screenCoordinate = cartesianToLinear(x, y - 1, screenWidth);
	output[screenCoordinate * 4] = 1.0f;
	output[screenCoordinate * 4 + 1] = 1.0f;
	output[screenCoordinate * 4 + 2] = 1.0f;
	output[screenCoordinate * 4 + 3] = 1.0f;
}

vector<vector<unsigned int>>
Drawings::extractConnectedVerticesCPU(const rapidjson::GenericValue<rapidjson::UTF8<>>::Array &meshDataDOM) {
	vector<vector<unsigned int >> connectedVertices;
	unsigned int vertexOffset = 0;

	for (int i = 0; i < meshDataDOM.Size(); ++i) { // Loop through every triangleMesh in scene
		auto curMesh = meshDataDOM[i].GetObject();

		// Loop through every face in triangleMesh
		for (auto &curMeshFaces : curMesh.FindMember(FACES)->value.GetArray()) {
			auto curMeshFaceVertices = curMeshFaces.GetObject().FindMember(VERTICES)->value.GetArray();
			for (int l = 0; l < curMeshFaceVertices.Size(); ++l) { // Loop through every vertex on face
				if (l == curMeshFaceVertices.Size() - 1) { // Make sure to add closing connection
					connectedVertices.push_back(
						{vertexOffset + curMeshFaceVertices[0].GetUint(),
						 vertexOffset + curMeshFaceVertices[l].GetUint()});
				} else {
					connectedVertices.push_back({vertexOffset + curMeshFaceVertices[l].GetUint(),
					                             vertexOffset + curMeshFaceVertices[l + 1].GetUint()});
				}
			}
		}

		// Increment vertex index offset with this completed triangleMesh
		vertexOffset += curMesh.FindMember(VERTICES)->value.GetArray().Size();
	}
	return connectedVertices;
}

void Drawings::drawWireframeCPU(float screenWidth, float screenHeight, float *pixData, const float *screenCoordinates,
                                vector<vector<unsigned int>> &connectedVertices) {
	for (auto &connection : connectedVertices) {
		// Get vertices
		float tar[] = {screenCoordinates[connection[0] * 3],
		               screenCoordinates[connection[0] * 3 + 1],
		               screenCoordinates[connection[0] * 3 + 2]};
		float src[] = {screenCoordinates[connection[1] * 3],
		               screenCoordinates[connection[1] * 3 + 1],
		               screenCoordinates[connection[1] * 3 + 2]};

		// Skip if was also skipped during conversion (left as -1 in conversion)
		if (tar[0] == -1 || src[0] == -1) {
			cout << "Skipping divide by 0" << endl;
			continue;
		}

		// get direction vector
		float dirX = tar[0] - src[0];
		float dirY = tar[1] - src[1];

		// calculate normalized vector
		float mag = sqrt(dirX * dirX + dirY * dirY);
		if (mag == 0) { // skip if the points have no delta
			continue;
		}
		float normX = dirX / mag;
		float normY = dirY / mag;

		// draw points while moving along
		float drawX = src[0];
		float drawY = src[1];

		// keep track of how far you have left
		int drawXDelta;
		int drawYDelta;

		do {
			drawX += normX;
			drawY += normY;
			drawXDelta = (int) round(abs(tar[0] - drawX));
			drawYDelta = (int) round(abs(tar[1] - drawY));

			if (drawX > screenWidth || drawX < 0 || drawY > screenHeight ||
			    drawY < 0) {
				break;
			}
			Drawings::drawDotCPU(drawX, drawY, pixData, screenWidth);
		} while (drawXDelta >= 3 || drawYDelta >= 3);
	}
}
