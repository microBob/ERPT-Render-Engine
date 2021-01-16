#include "include/main.h"
#include "include/kernels.cuh"
#include "include/communication.h"
#include "include/transformations.cuh"

int main() {
	//// SECTION: Variables and instances
	Communication com;
	Transformations transformations;


	//// SECTION: Connect to addon and read in data
	if (!com.ConnectSocket()) {
		return -1;
	}

	Document renderDataDOM = com.ReceiveData();
	assert(renderDataDOM.HasMember(SCENE));
	assert(renderDataDOM[SCENE].IsObject());


	//// SECTION: Setup pixData
	float *pixData;

	// Extract resolution
	assert(renderDataDOM[RESOLUTION].IsArray());
	const Value &resolutionData = renderDataDOM[RESOLUTION];
	assert(resolutionData[0].IsNumber());
	assert(resolutionData[1].IsNumber());

	size_t pixDataSize = resolutionData[0].GetFloat() * resolutionData[1].GetFloat() * 4 *
	                     sizeof(float); // Assume 1080 in case of read failure

	cudaMallocManaged(&pixData, pixDataSize);
	fill_n(pixData, pixDataSize / sizeof(float), 1.0f);


	//// SECTION: Convert to Camera space
	/// Create camera matrix
	// Verify camera data exists
	assert(renderDataDOM[SCENE][CAMERA].IsObject());
	// Verify location data exists
	assert(renderDataDOM[SCENE][CAMERA][LOCATION].IsArray());
	const Value &cameraLocation = renderDataDOM[SCENE][CAMERA][LOCATION];
	for (int i = 0; i < 3; ++i) {
		assert(cameraLocation[i].IsNumber());
	}
	// Verify rotation data exists
	assert(renderDataDOM[SCENE][CAMERA][ROTATION].IsArray());
	const Value &cameraRotation = renderDataDOM[SCENE][CAMERA][ROTATION];
	for (int i = 0; i < 3; ++i) {
		assert(cameraRotation[i].IsNumber());
	}
	// Once all Verified, set translation matrix
	transformations.set_worldToCameraMatrix(cameraLocation[0].GetFloat(), cameraLocation[1].GetFloat(),
	                                        cameraLocation[2].GetFloat(), cameraRotation[0].GetFloat(),
	                                        cameraRotation[1].GetFloat(), cameraRotation[2].GetFloat());

	/// Decompose mesh data into vertices
	



	//// SECTION: Convert and send data
	com.ConvertAndSend(pixData, pixDataSize);

	com.DisconnectSocket();
	cudaFree(pixData);
	return 0;
}
