#include "include/kernels.cuh"

int main() {
	//// SECTION: Socket Variables
	int sock;
	struct sockaddr_in server_address{};

	int floatPrecision = detectFloatPrecision();

	//// SECTION: Connect Socket
	cout << "Connecting to addon" << endl;

	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		cout << "Socket creation error" << endl;
		return -1;
	}

	server_address.sin_family = AF_INET;
	server_address.sin_port = htons(SOCKET_PORT);

	if (inet_pton(AF_INET, "127.0.0.1", &server_address.sin_addr) <= 0) {
		cout << "Invalid Address" << endl;
		return -1;
	}
	if (connect(sock, (struct sockaddr *) &server_address, sizeof(server_address)) < 0) {
		cout << "Connection Failed" << endl;
		return -1;
	}


	//// SECTION: Read in render data
	// TODO: infinite read in from socket
//	char resolutionInputBuffer[32] = {0};
//	int resolutionReadIn = read(sock, resolutionInputBuffer, 32);
//
//	if (resolutionReadIn > 0) { // Received something
//		// Parse input
//		char *resolution[2];
//		resolution[0] = strtok(resolutionInputBuffer, " ");
//		resolution[1] = strtok(nullptr, " ");
//		// Convert to int
//		unsigned int resolutionX = strtol(resolution[0], nullptr, 10);
//		unsigned int resolutionY = strtol(resolution[1], nullptr, 10);
//
//		pixDataSize = resolutionX * resolutionY * 4 * sizeof(float); // set pixDataSize based on input
//	} else if (resolutionReadIn == 0) {
//		cerr << "[ERROR]: EOF on reading resolution input" << endl;
//	} else {
//		cerr << "[ERROR]: On reading resolution reading input" << endl;
//	}

	//// SECTION: Setup pixData
	float *pixData;
	size_t pixDataSize = 1920 * 1080 * 4 * sizeof(float); // Assume 1080 in case of read failure

	//// SECTION: Convert and send data
	cudaMallocManaged(&pixData, pixDataSize);
	fill_n(pixData, pixDataSize / sizeof(float), 1.0f);

	cout << "Begin Conversion" << endl;
	auto beforeConvert = high_resolution_clock::now();

	StringBuffer json;
	Writer<StringBuffer> writer(json);

	writer.StartArray();
	for (int i = 0; i < pixDataSize / sizeof(float) / 4; ++i) {
		writer.StartArray();
		for (int j = 0; j < 4; ++j) {
			writer.Double(pixData[i * 4 + j]);
		}
		writer.EndArray();
	}
	writer.EndArray();

	auto afterConvert = high_resolution_clock::now();
	int convertDur = duration_cast<milliseconds>(afterConvert - beforeConvert).count();
	cout << "Conversion Duration: " << convertDur << endl;

	write(sock, json.GetString(), json.GetLength());

	auto afterSend = high_resolution_clock::now();
	int sendDur = duration_cast<milliseconds>(afterSend - afterConvert).count();

	cout << "Message sent! Took: " << sendDur << endl;

	close(sock);
	cudaFree(pixData);
	return 0;
}
