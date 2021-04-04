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

	unsigned int screenWidth, screenHeight;


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

	screenWidth = resolutionData[0].GetUint();
	screenHeight = resolutionData[1].GetUint();

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
	/// Translate scene data
	// Mesh
	TriangleMesh triangleMesh;
	auto meshDataDOM = sceneDataDOM.FindMember(MESHES)->value.GetObject();
	for (auto &curVertex : meshDataDOM.FindMember(VERTICES)->value.GetArray()) {
		auto vertexArray = curVertex.GetArray();
		triangleMesh.vertices.push_back(
			make_float3(vertexArray[0].GetFloat(), vertexArray[1].GetFloat(),
			            vertexArray[2].GetFloat()));
	}
	for (auto &curFace : meshDataDOM.FindMember(INDICES)->value.GetArray()) {
		auto indexArray = curFace.GetArray();
		triangleMesh.indices.push_back(
			make_uint3(indexArray[0].GetUint(), indexArray[1].GetUint(), indexArray[2].GetUint()));
	}

	// Camera
	auto cameraDataDOM = sceneDataDOM.FindMember(CAMERA)->value.GetObject();
	auto cameraLocation = cameraDataDOM.FindMember(LOCATION)->value.GetArray();
	auto cameraDirection = cameraDataDOM.FindMember(DIRECTION)->value.GetArray();
	auto cameraUp = cameraDataDOM.FindMember(UP)->value.GetArray();
	auto cameraFov = cameraDataDOM.FindMember(FOV)->value.GetFloat();

	Camera camera = {
		make_float3(cameraLocation[0].GetFloat(), cameraLocation[1].GetFloat(), cameraLocation[2].GetFloat()),
		make_float3(cameraDirection[0].GetFloat(), cameraDirection[1].GetFloat(), cameraDirection[2].GetFloat()),
		make_float3(cameraUp[0].GetFloat(), cameraUp[1].GetFloat(), cameraUp[2].GetFloat()),
		cameraFov
	};
	raytracing.setCamera(camera);

	// Screen / frame buffer size
	uint2 frameBufferSize = {screenWidth, screenHeight};
	raytracing.setFrameSize(frameBufferSize);

	/// Init OptiX
	try {
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
