#include "include/kernels.cuh"

int main() {
	int sock, readIn;
	struct sockaddr_in server_address{};

	char inputBuffer[1024] = {0};

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

	//// Connection worked and everything
	float *pixData;
	size_t pixDataSize = 1920 * 1080 * 4 * sizeof(float);
	cudaMallocManaged(&pixData, pixDataSize);
	pixData[0] = 1.83f;
	pixData[1] = 2.0f;

	cout << "Begin Conversion" << endl;

	auto beforeConvert = high_resolution_clock::now();

	stringstream ss;
	for (int i = 0; i < pixDataSize / sizeof(float); i++) {
		ss << pixData[i] << ' '; // 2 sec
	}

	auto afterConvert = high_resolution_clock::now();
	int convertDur = duration_cast<milliseconds>(afterConvert - beforeConvert).count();
	cout << "Conversion Duration: " << convertDur << endl;


//	send(sock, data.data(), data.size(), 0);
	send(sock, ss.str().c_str(), ss.str().length(), 0);

	auto afterSend = high_resolution_clock::now();
	int sendDur = duration_cast<milliseconds>(afterSend - afterConvert).count();

	cout << "Message sent! Took: " << sendDur << endl;

	/// Get input
//	readIn = read(sock, inputBuffer, 1024);
//	if (readIn > 0) {
//		cout << "Input: " << inputBuffer << endl;
//	} else if (readIn == 0) {
//		cout << "EOF" << endl;
//	} else {
//		cout << "Error reading input" << endl;
//	}

	close(sock);
	cudaFree(pixData);
	return 0;
}
