#include "include/main.h"
#include "include/kernels.cuh"
#include "include/communication.h"
#include "include/transformations.cuh"

int main() {
	//// SECTION: Variables and instances
	/// Class instances
	static Kernels k;
	Communication com;
	Transformations transformations(k);

	/// Major data variables
	Document renderDataDOM;

	float *sceneVertices;
	size_t sceneVerticesByteSize;


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

	size_t pixDataSize = resolutionData[0].GetFloat() * resolutionData[1].GetFloat() * 4 *
	                     sizeof(float); // Assume 1080 in case of read failure

	cudaMallocManaged(&pixData, pixDataSize);
	fill_n(pixData, pixDataSize / sizeof(float), 1.0f);


	//// SECTION: Convert to Camera space
	/// Create camera matrix
	// Get camera data and verify existence
	auto cameraDataDOM = sceneDataDOM.FindMember(CAMERA)->value.GetObject();
	// Verify location data exists
	auto cameraLocation = cameraDataDOM.FindMember(LOCATION)->value.GetArray();
	// Verify rotation data exists
	auto cameraRotation = cameraDataDOM.FindMember(ROTATION)->value.GetArray();
	// Once all Verified, set translation matrix
	transformations.set_worldToCameraMatrix(cameraLocation[0].GetFloat(), cameraLocation[1].GetFloat(),
	                                        cameraLocation[2].GetFloat(), cameraRotation[0].GetFloat(),
	                                        cameraRotation[1].GetFloat(), cameraRotation[2].GetFloat());

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
	sceneVerticesByteSize = 4 * rawVertices.size() * sizeof(float);
	cudaMallocManaged(&sceneVertices, sceneVerticesByteSize);
	// Transfer too CPU
	cudaMemAdvise(sceneVertices, sceneVerticesByteSize, cudaMemAdviseSetPreferredLocation, k.get_cpuID());
	// Set values
	sceneVertices = &rawVertices[0];
	// Switch to GPU
	cudaMemAdvise(sceneVertices, sceneVerticesByteSize, cudaMemAdviseSetPreferredLocation, k.get_gpuID());
	cudaMemAdvise(sceneVertices, sceneVerticesByteSize, cudaMemAdviseSetReadMostly, k.get_gpuID());
	cudaMemPrefetchAsync(sceneVertices, sceneVerticesByteSize, k.get_gpuID());



	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataSize);

	com.DisconnectSocket();
	cudaFree(pixData);
	return 0;
}
