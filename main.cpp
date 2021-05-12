#include "include/main.h"
#include "include/kernels.cuh"
#include "include/raytracing.h"
#include "include/communication.h"
#include "include/transformations.cuh"

extern "C" int main() {
	//// SECTION: Variables and instances
	/// Class instances
	Communication com;
	Transformations transformations;
	Raytracing raytracing;

	/// Major data variables
	Document renderDataDOM;
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


	//// SECTION: Setup OptiX
	/// Screen / frame buffer size
	uint2 frameBufferSize = {screenWidth, screenHeight};
	raytracing.setFrameSize(frameBufferSize);

	/// Translate scene data
	vector<TriangleMesh> triangleMeshes;
	// For each Mesh
	for (auto &curMesh : sceneDataDOM.FindMember(MESHES)->value.GetArray()) {
		TriangleMesh curMeshEncode;
		// Vertices
		for (auto &curVertex : curMesh.FindMember(VERTICES)->value.GetArray()) {
			auto vertexArray = curVertex.GetArray();
			curMeshEncode.vertices.push_back(
				make_float3(vertexArray[0].GetFloat(), vertexArray[1].GetFloat(),
				            vertexArray[2].GetFloat()));
		}
		// Face Indices
		for (auto &curFace : curMesh.FindMember(INDICES)->value.GetArray()) {
			auto indexArray = curFace.GetArray();
			curMeshEncode.indices.push_back(
				make_uint3(indexArray[0].GetUint(), indexArray[1].GetUint(), indexArray[2].GetUint()));
		}
		// Kind
		curMeshEncode.meshKind = static_cast<MeshKind>(curMesh.FindMember(KIND)->value.GetInt());
		// Color
		if (curMeshEncode.meshKind == Mesh) {
			curMeshEncode.color = {0.2f, 0.8f, 0.2f};
		}

		// Add to triangleMeshes
		triangleMeshes.push_back(curMeshEncode);
	}

	/// Camera
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

	/// Init OptiX
	try {
		raytracing.initOptix(triangleMeshes);
	} catch (runtime_error &error) {
		cout << error.what() << endl;
		exit(1);
	}

	//// SECTION: OptiX render
	raytracing.optixRender(10, 0);
	raytracing.downloadRender(pixData);


	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataByteSize);


	//// SECTION: Cleanup
	com.DisconnectSocket();
	cudaFree(pixData);
	transformations.cleanup();
	return 0;
}
