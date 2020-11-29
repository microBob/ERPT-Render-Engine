cmake_minimum_required(VERSION 3.17)
project(PythonAndC)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(../../External/pybind11 ../../External/pybind11/build)

set(PYTHON_EXECUTABLE /snap/blender/current/2.91/python/bin/python3.7m)

pybind11_add_module(${PROJECT_NAME} main.cpp)
