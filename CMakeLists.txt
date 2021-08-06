cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

SET(CMAKE_CXX_FLAGS "/openmp /Ot")

project(Volume LANGUAGES CXX CUDA)

enable_language(CUDA)

file(GLOB CORE_HEADERS *.h *.hpp *.cuh)
file(GLOB CORE_SOURCES *cpp *.cu)

source_group("Include" FILES ${CORE_HEADERS}) 
source_group("Source" FILES ${CORE_SOURCES})

include_directories(${PROJECT_SOURCE_DIR})
  
add_executable(MPM ${CORE_HEADERS} ${CORE_SOURCES})
set_target_properties(MPM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)