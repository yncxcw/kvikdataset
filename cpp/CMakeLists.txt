# =============================================================================
# Copyright (c) 2023, ynjassionchen@gmail.com
# =============================================================================
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

project(
    TarIOLib
    VERSION 1.0.0
    LANGUAGES CXX
)

# This needs to be customized based on your setting
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/nvcc")

add_library(
   kvikdataset_lib  INTERFACE 
)

# Defines the linking libraries
# From ChatGPT
find_package(Threads REQUIRED)
if(Threads_FOUND)
    message(STATUS "Threads package found:")
    message(STATUS "  Threads libraries: ${CMAKE_THREAD_LIBS_INIT}")
else()
    message(FATAL_ERROR "Threads package not found.")
endif()

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND) 
    message(STATUS "CUDA Toolkit found:")
    message(STATUS "  Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  Libraries: ${CUDAToolkit_LIBRARIES}")
    message(STATUS "  Include directories: ${CUDAToolkit_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "CUDA Toolkit not found.")
endif()

target_include_directories(kvikdataset_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kvikdataset_lib INTERFACE Threads::Threads)
target_link_libraries(kvikdataset_lib INTERFACE CUDA::toolkit)
target_link_libraries(kvikdataset_lib INTERFACE CUDA::cuFile)
target_link_libraries(kvikdataset_lib INTERFACE CUDA::cudart)
target_compile_features(kvikdataset_lib INTERFACE cxx_std_17)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE kvikdataset_lib)

add_subdirectory(test)