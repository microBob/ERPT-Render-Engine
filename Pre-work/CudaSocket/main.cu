#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define PORT 8083

using namespace std;

int main() {
	int sock, readIn;
	struct sockaddr_in server_address{};

	string message = "Hello from C++";
	char inputBuffer[1024] = {0};


	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		cout << "Socket creation error" << endl;
		return -1;
	}

	server_address.sin_family = AF_INET;
	server_address.sin_port = htons(PORT);

	if (inet_pton(AF_INET, "127.0.0.1", &server_address.sin_addr) <= 0) {
		cout << "Invalid Address" << endl;
		return -1;
	}
	if (connect(sock, (struct sockaddr *) &server_address, sizeof(server_address)) < 0) {
		cout << "Connection Failed" << endl;
		return -1;
	}

	//// Connection worked and everything
	if (send(sock, message.c_str(), message.length(), 0) < 0) {
		cout << "Message fail to send!" << endl;
		return -1;
	}

	cout << "Message sent!" << endl;

	/// Get input
	readIn = read(sock, inputBuffer, 1024);
	if (readIn > 0) {
		cout << "Input: " << inputBuffer << endl;
	} else if (readIn == 0) {
		cout << "EOF" << endl;
	} else {
		cout << "Error reading input" << endl;
	}

	return 0;
}
