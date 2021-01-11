cmake_minimum_required(VERSION 3.17)
project(ERPT_Render_Engine CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit)

add_executable(
	ERPT_Render_Engine
	main.cpp include/main.h
	include/constants.h
	include/communication.h src/communication.cpp
	include/kernels.cuh
	include/transformations.cuh src/transformations.cu src/kernels.cu)
add_executable(LABS_Convert labs/convertLabs.cpp)

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
	include
)
target_link_libraries(
	${PROJECT_NAME}
	PRIVATE
	CUDA::cudart
	CUDA::cublas
)