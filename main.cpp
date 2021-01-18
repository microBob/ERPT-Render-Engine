#include "include/main.h"
#include "include/kernels.cuh"
#include "include/communication.h"
#include "include/transformations.cuh"

unsigned int cartesianToLinear(float x, float y, float screenWidth, float screenHeight) {
	return (unsigned int) ((y + screenHeight / 2) * screenWidth + (x + screenWidth / 2));
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

	size_t pixDataSize = screenWidth * screenHeight * 4 *
	                     sizeof(float); // Assume 1080 in case of read failure

	cudaMallocManaged(&pixData, pixDataSize);
	cudaMemPrefetchAsync(pixData, pixDataSize, k.get_cpuID());
	fill_n(pixData, pixDataSize / sizeof(float), 0); // Fill with black screen


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
	                                             cameraRotation[1].GetFloat(), cameraRotation[2].GetFloat(), cameraFov,
	                                             screenWidth, screenHeight, cameraClipping[0].GetFloat(),
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
	cudaMemAdvise(perspectiveVertices, sceneVerticesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemPrefetchAsync(perspectiveVertices, sceneVerticesByteSize, k.get_gpuID());
	// Convert world to perspective
	transformations.convertWorldToPerspectiveSpace(sceneVertices, sceneVertexCount, perspectiveVertices);

	/// Convert perspective to screen coordinates
	float *screenCoordinates;
	size_t screenCoordinatesByteSize = 2 * sceneVertexCount * sizeof(float);
	// Initialize output
	cudaMallocManaged(&screenCoordinates, screenCoordinatesByteSize);
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_gpuID());
	// MemAdvise input
	cudaMemAdvise(perspectiveVertices, sceneVerticesByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(perspectiveVertices, sceneVerticesByteSize, k.get_gpuID());
	// Convert perspective to screen
	k.set_kernelThreadsAndBlocks(sceneVertexCount);
	transformations.convertPerspectiveToScreenSpace(perspectiveVertices, sceneVertexCount, screenWidth, screenHeight,
	                                                screenCoordinates);
	cudaFree(perspectiveVertices); // Get rid of perspectiveVertices after convert to screen

	//// SECTION: Draw point cloud
	/// Switch screenCoordinates to CPU
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetReadMostly, k.get_cpuID());
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_cpuID());
	for (int i = 0; i < sceneVertexCount; ++i) {
		unsigned int screenCoordinate = cartesianToLinear(screenCoordinates[i * 2], screenCoordinates[i * 2 + 1],
		                                                  screenWidth, screenHeight);
		pixData[screenCoordinate] = 1.0f;
		pixData[screenCoordinate + 1] = 1.0f;
		pixData[screenCoordinate + 2] = 1.0f;
		pixData[screenCoordinate + 3] = 1.0f;
	}


	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataSize);


	//// SECTION: Cleanup
	com.DisconnectSocket();
	cudaFree(pixData);
	cudaFree(sceneVertices);
	transformations.cleanup();
	return 0;
}
