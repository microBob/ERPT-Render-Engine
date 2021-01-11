//
// Created by microbobu on 1/10/21.
//
#include "../include/kernels.cuh"

int Kernels::get_gpuID() const {
	return gpuID;
}

int Kernels::get_cpuID() const {
	return cpuID;
}