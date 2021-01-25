#include "include/main.h"
#include "include/kernels.cuh"
#include "include/communication.h"
#include "include/transformations.cuh"

unsigned int cartesianToLinear(float x, float y, float screenWidth) {
	return (unsigned int) (round(y) * screenWidth + round(x));
}

void drawDot(float x, float y, float *output, float screenWidth) {
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

int main() {
	//// SECTION: Variables and instances
	/// Class instances
	Communication com;
	Transformations transformations;

	/// Major data variables
	Document renderDataDOM;

	float *sceneVertices;
	size_t sceneVerticesByteSize;

	float screenWidth, screenHeight;


	//// SECTION: Connect to addon and read in data
	if (!com.ConnectSocket()) {
		return -1;
	}

	renderDataDOM = com.ReceiveData();
	// Get scene data and verify existence
	auto sceneDataDOM = renderDataDOM.FindMember(SCENE)->value.GetObject();


	//// SECTION: Setup pixData
	float *pixData;

	// Extract resolution
	auto resolutionData = renderDataDOM.FindMember(RESOLUTION)->value.GetArray();

	screenWidth = resolutionData[0].GetFloat();
	screenHeight = resolutionData[1].GetFloat();

	size_t pixDataByteSize = screenWidth * screenHeight * 4 * sizeof(float);

	cudaMallocManaged(&pixData, pixDataByteSize);
	cudaMemPrefetchAsync(pixData, pixDataByteSize, k.get_cpuID());
//	fill_n(pixData, pixDataByteSize, 0);
	for (int i = 0; i < pixDataByteSize / sizeof(float); ++i) { // Fill pixData with a black screen
		if ((i + 1) % 4 == 0) { // Set alpha to 1
			pixData[i] = 1;
		} else { // Set everything else to 0
			pixData[i] = 0;
		}
	}


	//// SECTION: Convert Data to screen space
	/// Create camera matrix
	// Get camera data and verify existence
	auto cameraDataDOM = sceneDataDOM.FindMember(CAMERA)->value.GetObject();
	auto cameraLocation = cameraDataDOM.FindMember(LOCATION)->value.GetArray();
	auto cameraRotation = cameraDataDOM.FindMember(ROTATION)->value.GetArray();
	auto cameraFov = cameraDataDOM.FindMember(FOV)->value.GetFloat();
	auto cameraClipping = cameraDataDOM.FindMember(CLIP)->value.GetArray();
	// Once all Verified, set translation matrix
	transformations.set_worldToPerspectiveMatrix(cameraLocation[0].GetFloat(), cameraLocation[1].GetFloat(),
	                                             cameraLocation[2].GetFloat(), cameraRotation[0].GetFloat(),
	                                             cameraRotation[1].GetFloat(), cameraRotation[2].GetFloat(),
	                                             cameraFov, screenWidth / screenHeight,
	                                             cameraClipping[0].GetFloat(),
	                                             cameraClipping[1].GetFloat());

	/// Decompose mesh data into vertices
	auto meshDataDOM = sceneDataDOM.FindMember(MESHES)->value.GetArray();
	vector<float> rawVertices; // Utilize vector because of unknown vertices count

	// Loop through every mesh in data
	for (auto &mesh : meshDataDOM) {
		auto meshVertices = mesh.FindMember(VERTICES)->value.GetArray(); // Get vertices from mesh
		for (auto &vertex : meshVertices) { // Loop through each vertex
			for (auto &coordinates : vertex.GetArray()) { // Loop through XYZ of vertex
				rawVertices.push_back(coordinates.GetFloat());
			}
			rawVertices.push_back(1); // Add extra 1 to complete 4D vector
		}
	}

	// Malloc vertices data
	sceneVerticesByteSize = rawVertices.size() * sizeof(float);
	cudaMallocManaged(&sceneVertices, sceneVerticesByteSize);
	// Transfer to CPU
	cudaMemPrefetchAsync(sceneVertices, sceneVerticesByteSize, k.get_cpuID());
	// Convert values from vector to array
	copy(rawVertices.begin(), rawVertices.end(), sceneVertices);
	// Switch to GPU
	cudaMemAdvise(sceneVertices, sceneVerticesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemAdvise(sceneVertices, sceneVerticesByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(sceneVertices, sceneVerticesByteSize, k.get_gpuID());

	int sceneVertexCount = (int) rawVertices.size() / 4;
	/// Convert vertices from world to perspective
	float *perspectiveVertices;
	// Initialize output
	cudaMallocManaged(&perspectiveVertices, sceneVerticesByteSize);
	cudaMemPrefetchAsync(perspectiveVertices, sceneVerticesByteSize, k.get_cpuID());
	fill_n(perspectiveVertices, sceneVerticesByteSize, 0);
	cudaMemAdvise(perspectiveVertices, sceneVerticesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemPrefetchAsync(perspectiveVertices, sceneVerticesByteSize, k.get_gpuID());
	// Convert world to perspective
	transformations.convertWorldToPerspectiveSpace(sceneVertices, sceneVertexCount, perspectiveVertices);

	/// Convert perspective to screen coordinates
	float *screenCoordinates;
	size_t screenCoordinatesByteSize = 3 * sceneVertexCount * sizeof(float);
	// Initialize output
	cudaMallocManaged(&screenCoordinates, screenCoordinatesByteSize);
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_cpuID());
	fill_n(screenCoordinates, screenCoordinatesByteSize, -1);
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_gpuID());
	// MemAdvise input
	cudaMemAdvise(perspectiveVertices, sceneVerticesByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(perspectiveVertices, sceneVerticesByteSize, k.get_gpuID());
	// Convert perspective to screen
//	k.set_kernelThreadsAndBlocks(sceneVertexCount);

//	for (int i = 0; i < sceneVertexCount * 4; ++i) {
//		cout << perspectiveVertices[i];
//		if ((i + 1) % 4 == 0) {
//			cout << endl;
//		} else {
//			cout << ",\t";
//		}
//	}
//	cout << endl;
//	for (int i = 0; i < sceneVertexCount * 3; ++i) {
//		cout << screenCoordinates[i];
//		if ((i + 1) % 3 == 0) {
//			cout << endl;
//		} else {
//			cout << ",\t";
//		}
//	}
//	cout << endl;

	for (int i = 0; i < sceneVertexCount; ++i) {
		if (abs(perspectiveVertices[i * 4 + 2]) > 1) {
			cout << "Skipping outside" << endl;
			continue;
		}
		// Skip if point will cause divide by 0
		if (perspectiveVertices[i * 4 + 3] == 0) {
			cout << "Skipping div by 0" << endl;
			continue;
		}

		screenCoordinates[i * 3] =
			(perspectiveVertices[i * 4] / perspectiveVertices[i * 4 + 3] + 1) * screenWidth / 2;
		screenCoordinates[i * 3 + 1] =
			(perspectiveVertices[i * 4 + 1] / perspectiveVertices[i * 4 + 3] + 1) * screenHeight / 2;
		screenCoordinates[i * 3 + 2] = perspectiveVertices[i * 4 + 3];
	}

	cudaFree(perspectiveVertices); // Get rid of perspectiveVertices after convert to screen

	//// SECTION: Draw Wireframe
	/// Switch screenCoordinates to CPU
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetReadMostly, k.get_cpuID());
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_cpuID());
	for (int i = 0; i < sceneVertexCount; ++i) {
		cout << screenCoordinates[i * 3] << ",\t" << screenCoordinates[i * 3 + 1] << ",\t"
		     << screenCoordinates[i * 3 + 2]
		     << endl;
	}
	cout << endl << endl;

	vector<vector<unsigned int >> connectedVertices;
	for (int i = 0; i < meshDataDOM.Size(); ++i) {
		auto curMesh = meshDataDOM[i].GetObject();

		unsigned int vertexOffset = 0;
		for (int j = 0; j < i; ++j) {
			vertexOffset += meshDataDOM[j].GetObject().FindMember(VERTICES)->value.GetArray().Size();
		}

		for (auto &curMeshFaces : curMesh.FindMember(FACES)->value.GetArray()) {
			auto curMeshFaceVertices = curMeshFaces.GetObject().FindMember(VERTICES)->value.GetArray();
			for (int l = 0; l < curMeshFaceVertices.Size(); ++l) {
				if (l == curMeshFaceVertices.Size() - 1) {
					connectedVertices.push_back(
						{curMeshFaceVertices[vertexOffset].GetUint(), curMeshFaceVertices[vertexOffset + l].GetUint()});
				} else {
					connectedVertices.push_back({curMeshFaceVertices[vertexOffset + l].GetUint(),
					                             curMeshFaceVertices[vertexOffset + l + 1].GetUint()});
				}
			}
		}
	}

	for (auto &connection : connectedVertices) {
		// Get vertices
		float tar[] = {screenCoordinates[connection[0] * 3],
		               screenCoordinates[connection[0] * 3 + 1],
		               screenCoordinates[connection[0] * 3 + 2]};
		float src[] = {screenCoordinates[connection[1] * 3],
		               screenCoordinates[connection[1] * 3 + 1],
		               screenCoordinates[connection[1] * 3 + 2]};

		// Skip if was also skipped during conversion
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
			drawDot(drawX, drawY, pixData, screenWidth);
		} while (drawXDelta >= 3 || drawYDelta >= 3);
	}


	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataByteSize);


	//// SECTION: Cleanup
	com.DisconnectSocket();
	cudaFree(pixData);
	cudaFree(sceneVertices);
	transformations.cleanup();
	return 0;
}
