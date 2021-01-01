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
	cout << "Reading in data" << endl;

	vector<char> dataBuffer; // Render data buffer

	// Data buffer size needed calculation
	unsigned long dataSize = -1;
	int indexOfDataStart;
	unsigned long dataReadIn = 0;

	do {
		char smallDataBuffer[1024];
		int dataIn = read(sock, smallDataBuffer, 1024);

		if (dataSize == -1) {
			indexOfDataStart = int(strchr(smallDataBuffer, '{') - smallDataBuffer);
			dataSize = 0;
			for (unsigned long i = 0; i < indexOfDataStart; ++i) {
				int digitConvert = (int) smallDataBuffer[i] - 48;
				dataSize += digitConvert * (unsigned long) pow(10, i);
			}
		}

		if (dataIn > 0) {
			for (int i = (dataReadIn == 0) ? indexOfDataStart : 0; i < dataIn; ++i) {
				dataReadIn++;
				dataBuffer.push_back(smallDataBuffer[i]);
			}
		} else if (dataIn == 0) {
			cerr << "[ERROR]: EOF on reading in render data" << endl;
			break;
		} else {
			cerr << "[ERROR]: on reading in render data" << endl;
			break;
		}
	} while (dataReadIn + 1 < dataSize);

	string dataString(dataBuffer.begin(), dataBuffer.end());

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
