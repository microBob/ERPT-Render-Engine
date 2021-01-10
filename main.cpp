#include "include/main.h"
#include "include/kernels.cuh"
#include "include/communication.h"

int main() {
	//// SECTION: Connect to addon and read in data
	if (!connectSocket()) {
		return -1;
	}

	Document renderDataDOM = receiveData();

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

	//// SECTION: Convert and send data
	convertAndSend(pixData, pixDataSize);

	disconnectSocket();
	cudaFree(pixData);
	return 0;
}
