cmake_minimum_required(VERSION 3.17)
project(ERPT_Render_Engine CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH
	"${CMAKE_SOURCE_DIR}/cmake"
	${CMAKE_MODULE_PATH})

find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED VERSION 7.0)

add_executable(
	ERPT_Render_Engine
	main.cpp include/main.h
	include/constants.h
	include/communication.h src/communication.cpp
	include/kernels.cuh
	include/transformations.cuh src/transformations.cu src/kernels.cu include/drawings.cuh src/drawings.cu include/raytracing.h src/raytracing.cpp)
add_executable(LABS labs/labs.cpp labs/another.h)

# For project
set_target_properties(
	${PROJECT_NAME}
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(
	${PROJECT_NAME}
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
	-arch=sm_61
	-gencode arch=compute_52,code=sm_52>
)
target_include_directories(
	${PROJECT_NAME}
	PRIVATE
	include
	${OptiX_INCLUDE}
)
target_link_libraries(
	${PROJECT_NAME}
	PRIVATE
	CUDA::cudart
	CUDA::cublas
)

# Copy for Labs
set_target_properties(
	LABS
	PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(
	LABS
	PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:
	-gencode arch=compute_61,code=sm_61>
)
target_include_directories(
	LABS
	PRIVATE
	include
)
target_link_libraries(
	LABS
	PRIVATE
	CUDA::cudart
	CUDA::cublas
)