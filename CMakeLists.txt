cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
 
include(CTest)

add_executable(cuda_test main.cu)

set_target_properties(cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
# add the executable
#add_executable(cmake_and_cuda main.cpp)
 
