#include <pybind11/pybind11.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

int add(int i, int j) {
	return i + j;
}

float sum(int j) {
	float *x;
	float result = 0.0f;
	cudaMallocManaged(&x, j * sizeof(float));

	for (int i = 0; i < j; ++i) {
		x[j] = (float) j;
	}

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasStatus_t status = cublasSasum(handle, j, x, 1, &result);

	cudaDeviceSynchronize();

	cudaFree(x);
	cublasDestroy(handle);

	if (status != CUBLAS_STATUS_SUCCESS) {
		return -1.0;
	}
	return result;
}

PYBIND11_MODULE(PythonAndC, m) {
	m.doc() = "pybind11 \"PythonAndC\" example";
	m.def("add", &add, "A function that adds two integers");
	m.def("sum", &sum, "A function that sums an array of length j");
}