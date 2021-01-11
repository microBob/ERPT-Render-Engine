//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_COMMUNICATION_H
#define ERPT_RENDER_ENGINE_COMMUNICATION_H

//// SECTION: Includes
#include "main.h"
#include <vector>
/// Socket
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>

/// RapidJSON
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


//// SECTION: Class definition
class Communication {
private:
	int sock;
	struct sockaddr_in server_address{};
public:
	bool ConnectSocket(); // Return true on success
	void DisconnectSocket() const;

	Document ReceiveData() const; // Return data from addon as a DOM
	void ConvertAndSend(float *pixData, size_t pixDataSize) const;
};

#endif //ERPT_RENDER_ENGINE_COMMUNICATION_H
