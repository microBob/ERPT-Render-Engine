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



	//// SECTION: Setup OptiX
	/// Read Scene
	TriangleMesh triangleMesh;

	// Setup camera
	auto cameraDataDOM = sceneDataDOM.FindMember(CAMERA)->value.GetObject();
	auto cameraLocation = cameraDataDOM.FindMember(LOCATION)->value.GetArray();
	auto cameraDirection = cameraDataDOM.FindMember(DIRECTION)->value.GetArray();
	auto cameraUp = cameraDataDOM.FindMember(UP)->value.GetArray();
	auto cameraFov = cameraDataDOM.FindMember(FOV)->value.GetFloat();

	Camera camera = {
		make_float3(cameraLocation[0].GetFloat(), cameraLocation[1].GetFloat(), cameraLocation[2].GetFloat()),
		make_float3(cameraDirection[0].GetFloat(), cameraDirection[1].GetFloat(), cameraDirection[2].GetFloat()),
		make_float3(cameraUp[0].GetFloat(), cameraUp[1].GetFloat(), cameraUp[2].GetFloat())
	};

	try {

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

	//// SECTION: OptiX render
	raytracing.optixRender();
	raytracing.downloadRender(pixData);


	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataByteSize);


	//// SECTION: Cleanup
	com.DisconnectSocket();
	cudaFree(pixData);
	transformations.cleanup();
	return 0;
}
