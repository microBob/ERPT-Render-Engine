#include "include/main.h"
#include "include/kernels.cuh"
#include "include/raytracing.h"
#include "include/communication.h"
#include "include/transformations.cuh"
#include "include/drawings.cuh"

extern "C" int main() {
	//// SECTION: Variables and instances
	/// Class instances
	Communication com;
	Transformations transformations;
	Raytracing raytracing;

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


	//// SECTION: Setup pixData and OptiX frame buffer
	/// pixData
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



	//// SECTION: Initialize OptiX
	try {
		TriangleMesh triangleMesh;
		/// Launch Parameters
		// Frame size
		int2 frameBufferSize = {static_cast<int>(screenWidth), static_cast<int>(screenHeight)};
		raytracing.setFrameSize(frameBufferSize);

		// Init OptiX
		raytracing.initOptix(triangleMesh);

	} catch (runtime_error &error) {
		cout << error.what() << endl;
		exit(1);
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
	Transformations::convertPerspectiveToScreenSpaceCPU(screenWidth, screenHeight, sceneVertexCount,
	                                                    perspectiveVertices,
	                                                    screenCoordinates);

	cudaFree(perspectiveVertices); // Get rid of perspectiveVertices after convert to screen


	//// SECTION: Draw Wireframe
	/// Switch screenCoordinates to CPU
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());
	cudaMemAdvise(screenCoordinates, screenCoordinatesByteSize, cudaMemAdviseSetReadMostly, k.get_cpuID());
	cudaMemPrefetchAsync(screenCoordinates, screenCoordinatesByteSize, k.get_cpuID());

	/// Extract connected vertices
//	vector<vector<unsigned int>> connectedVertices = Drawings::extractConnectedVerticesCPU(meshDataDOM);

	/// Draw edges between vertices
//	Drawings::drawWireframeCPU(screenWidth, screenHeight, pixData, screenCoordinates, connectedVertices);


	//// SECTION: OptiX render
	raytracing.optixRender();
	raytracing.downloadRender(pixData);


	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataByteSize);


	//// SECTION: Cleanup
	com.DisconnectSocket();
	cudaFree(pixData);
	cudaFree(sceneVertices);
	transformations.cleanup();
	return 0;
}
