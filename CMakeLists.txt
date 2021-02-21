cmake_minimum_required(VERSION 3.17)
project(ERPT_Render_Engine CUDA CXX)

# Set language standard
set(CMAKE_CXX_STANDARD 14)

# Add FindOptiX and other cmake files
set(CMAKE_MODULE_PATH
	"${CMAKE_SOURCE_DIR}/cmake"
	${CMAKE_MODULE_PATH})

# Import CUDA and OptiX
find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED VERSION 7.0)
find_program(BIN2C bin2c)
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file})
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
	  OUTPUT ${embedded_file}
	  COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
	  DEPENDS ${ptx_file}
	  COMMENT "compiling (and embedding ptx from) ${cuda_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()
# Convert device program into PTX
cuda_compile_and_embed(embeddedPtxCode src/optixDevicePrograms.cu)

# Add in code files
add_executable(
	${PROJECT_NAME}
	main.cpp include/main.h
	include/constants.h
	include/communication.h src/communication.cpp
	include/kernels.cuh
	include/transformations.cuh src/transformations.cu src/kernels.cu include/drawings.cuh src/drawings.cu include/raytracing.h src/raytracing.cpp src/optixDevicePrograms.cu
	${embeddedPtxCode}
)
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
	cuda
	cudart
	cublas
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
	cudart
	cublas
)