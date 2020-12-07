cmake_minimum_required(VERSION 3.17)
project(PythonAndC CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(../../External/pybind11 ../../External/pybind11/build)

pybind11_add_module(${PROJECT_NAME} main.cpp)

set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(
	${PROJECT_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
	-gencode arch=compute_61,code=sm_61>
)
target_include_directories(
	${PROJECT_NAME}
	PRIVATE
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
	${PROJECT_NAME}
	PRIVATE
	"/usr/lib/cuda-10.2/lib64/libcublas.so"
	"/usr/lib/cuda-10.2/lib64/libcudart.so"
#	-lcublas
)
