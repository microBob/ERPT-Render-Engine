cmake_minimum_required(VERSION 3.17)
project(CudaSocket CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaSocket main.cu)

set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON)
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
	-lcublas
)