cmake_minimum_required(VERSION 3.17)
project(PythonAndC)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(../../External/pybind11 ../../External/pybind11/build)

pybind11_add_module(${PROJECT_NAME} main.cpp)

#target_compile_options(
#	${PROJECT_NAME}
#	PRIVATE
#	-DPYTHON_EXECUTABLE=/snap/blender/51/2.91/python/bin/python3.7m
#)
#add_compile_options(-DPYTHON_EXECUTABLE=/snap/blender/51/2.91/python/bin/python3.7m)
