//
// Created by microbobu on 12/6/20.
//

#ifndef ERPT_RENDER_ENGINE_KERNELS_CUH
#define ERPT_RENDER_ENGINE_KERNELS_CUH

//// Includes
/// Main and Cuda
#include <cuda_runtime.h>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <chrono>
/// Sockets
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>


//// Namespaces
using namespace std;
using namespace chrono;


//// Defines and Macros
#define SOCKET_PORT 8083


//// Models


//// Kernel Functions


//// CPU Functions
int detectFloatPrecision();
#endif //ERPT_RENDER_ENGINE_KERNELS_CUH
