cmake_minimum_required(VERSION 3.17)
project(ERPT_Render_Engine CUDA CXX C)

# Set language standard
set(CMAKE_CXX_STANDARD 14)

# Add FindOptiX and other cmake files
set(CMAKE_MODULE_PATH
	"${CMAKE_SOURCE_DIR}/cmake"
	${CMAKE_MODULE_PATH})

# Get libraries
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED) # To compile ptx code
find_package(OptiX REQUIRED VERSION 7.0)
include_directories(${OptiX_INCLUDE}) # Needed general include for ptx compile
find_program(BIN2C bin2c)

# OptiX device program compile
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file})
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  #  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
  add_custom_command(
	  OUTPUT ${embedded_file}
	  COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
	  DEPENDS ${ptx_file}
	  COMMENT "compiling (and embedding ptx from) ${cuda_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()

cuda_compile_and_embed(embeddedPtxCode src/optixDevicePrograms.cu)

add_executable(
	${PROJECT_NAME}
	include/main.h main.cpp
	include/constants.h
	include/communication.h src/communication.cpp
	include/kernels.cuh src/kernels.cu
	include/transformations.cuh src/transformations.cu
	include/drawings.cuh src/drawings.cu
	include/raytracing.h src/raytracing.cpp
	include/optixLaunchParameters.h
	include/CUDABuffer.h
	${embeddedPtxCode}
)
add_executable(LABS labs/labs.cpp labs/another.h)

# For project
add_definitions(-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1)
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
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(
	${PROJECT_NAME}
	PRIVATE
	${optix_LIBRARY}
	CUDA::cudart
	CUDA::cuda_driver
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
	CUDA::cuda_driver
	CUDA::cublas
)