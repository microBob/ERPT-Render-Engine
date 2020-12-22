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

	//// SECTION: Convert and send data
	/// Setup pixData
	float *pixData;
	size_t pixDataSize = 1920 * 1080 * 4 * sizeof(float); // Assume 1080 in case of read failure

	char resolutionInputBuffer[32] = {0};
	int resolutionReadIn = read(sock, resolutionInputBuffer, 32);

	if (resolutionReadIn > 0) { // Received something
		cout << "Input: " << resolutionInputBuffer << endl;

		// Parse input
		char *resolution[2];
		resolution[0] = strtok(resolutionInputBuffer, " ");
		resolution[1] = strtok(nullptr, " ");
		// Convert to int
		unsigned int resolutionX = strtol(resolution[0], nullptr, 10);
		unsigned int resolutionY = strtol(resolution[1], nullptr, 10);

		pixDataSize = resolutionX * resolutionY * 4 * sizeof(float); // set pixDataSize based on input
	} else if (resolutionReadIn == 0) {
		cerr << "[ERROR]: EOF on reading resolution input" << endl;
	} else {
		cerr << "[ERROR]: On reading resolution reading input" << endl;
	}

	cudaMallocManaged(&pixData, pixDataSize);
	fill_n(pixData, pixDataSize / sizeof(float), 1.0f);

	cout << "Begin Conversion" << endl;

	auto beforeConvert = high_resolution_clock::now();

	stringstream ss;
	ss << '['; // Start with array open
	for (int i = 0; i < pixDataSize / sizeof(float) / 4; ++i) {
		ss << '[';
		for (int j = 0; j < 4; ++j) { // Write in python eval
			ss << "float(" << (int) (pixData[i] * pow(10, floatPrecision)) << ")/(10**" << floatPrecision << "),";
		}
		ss << "],";
	}
	ss << ']';

	auto afterConvert = high_resolution_clock::now();
	int convertDur = duration_cast<milliseconds>(afterConvert - beforeConvert).count();
	cout << "Conversion Duration: " << convertDur << endl;

	write(sock, ss.str().c_str(), ss.str().length());

	auto afterSend = high_resolution_clock::now();
	int sendDur = duration_cast<milliseconds>(afterSend - afterConvert).count();

	cout << "Message sent! Took: " << sendDur << endl;

	close(sock);
	cudaFree(pixData);
	return 0;
}
