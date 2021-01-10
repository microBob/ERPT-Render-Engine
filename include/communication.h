//
// Created by microbobu on 1/10/21.
//

#ifndef ERPT_RENDER_ENGINE_COMMUNICATION_H
#define ERPT_RENDER_ENGINE_COMMUNICATION_H

//// SECTION: Includes
#include "main.h"
/// Socket
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>

/// RapidJSON
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


//// SECTION: Methods
bool connectSocket(); // Return true on success
Document receiveData(); // Return data from addon as a DOM
void convertAndSend(float *pixData, size_t pixDataSize);
void disconnectSocket();


#endif //ERPT_RENDER_ENGINE_COMMUNICATION_H
