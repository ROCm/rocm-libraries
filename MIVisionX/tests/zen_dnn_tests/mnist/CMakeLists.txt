################################################################################
#
# MIT License
#
# Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

# Set Default Compiler & Standard
# aocc-linux-compiler -- ENV:ZENDNN_AOCC_COMP_PATH
if(NOT DEFINED ZENDNN_AOCC_COMP_PATH)
    if(NOT DEFINED ENV{ZENDNN_AOCC_COMP_PATH})
        message(FATAL_ERROR "ZENDNN_AOCC_COMP_PATH NOT FOUND -- aocc-compiler-X.X.XX install path must be set to ZENDNN_AOCC_COMP_PATH${ColourReset}")
    else()
        set(ZENDNN_AOCC_COMP_PATH $ENV{ZENDNN_AOCC_COMP_PATH} CACHE PATH "Path to which ZENDNN_AOCC_COMP_PATH has been installed")
    endif()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_COMPILER ${ZENDNN_AOCC_COMP_PATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${ZENDNN_AOCC_COMP_PATH}/bin/clang++)

project(zendnn_mnist_f32)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# -fopenmp -- Enable OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)

# aocl-linux-aocc -- ENV:ZENDNN_BLIS_PATH
if(NOT DEFINED ZENDNN_BLIS_PATH)
    if(NOT DEFINED ENV{ZENDNN_BLIS_PATH})
        message(FATAL_ERROR "ZENDNN_BLIS_PATH NOT FOUND -- aocl-linux-aocc-X.X.XX install path must be set to ZENDNN_BLIS_PATH${ColourReset}")
    else()
        set(ZENDNN_BLIS_PATH $ENV{ZENDNN_BLIS_PATH} CACHE PATH "Path to which ZENDNN_BLIS_PATH has been installed")
    endif()
endif()

# amd Zen DNN Git Source
if(NOT DEFINED ZENDNN_GIT_ROOT)
    if(NOT DEFINED ENV{ZENDNN_GIT_ROOT})
        message(FATAL_ERROR "ZENDNN_GIT_ROOT NOT FOUND -- ZEN DNN Git Source path must be set to ZENDNN_GIT_ROOT${ColourReset}")
    else()
        set(ZENDNN_GIT_ROOT $ENV{ZENDNN_GIT_ROOT} CACHE PATH "Path to which ZENDNN_GIT_ROOT has been installed")
    endif()
endif()

# set global compiler flags for the project
# -O3 -- Optimize output file
# -fPIC -- Generate position-independent code if possible.
# -march -- Generate code for given CPU. [znver2 -- Zen Version 2]
# -Wreturn-type -- Warn whenever a function's return type defaults to "int" (C), or about inconsistent return types (C++).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLIBM_ENABLE=1 -DBIAS_ENABLED=1 -DZENDNN_ENABLE=1 -DZENDNN_X64=1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -march=znver2 -Wreturn-type")

# INCLUDE FILES
# ZENDNN_GIT_ROOT includes
include_directories(${ZENDNN_GIT_ROOT}/inc ${ZENDNN_GIT_ROOT}/src ${ZENDNN_GIT_ROOT}/src/common ${ZENDNN_GIT_ROOT}/src/cpu)
# ZENDNN_BLIS_PATH includes
include_directories(${ZENDNN_BLIS_PATH}/include)
# GNU Includes
include_directories(/usr/lib/x86_64-linux-gnu/include)
# test utilities
include_directories(${ZENDNN_GIT_ROOT}/tests/api_tests)

# Link Library Directories
link_directories(${ZENDNN_GIT_ROOT}/_out/lib)
link_directories(${ZENDNN_BLIS_PATH}/lib)
link_directories(/usr/lib/x86_64-linux-gnu/lib)

file(GLOB My_Source_Files ./source/*.cpp)
add_executable(${PROJECT_NAME} ${My_Source_Files})

set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} amdZenDNN blis-mt alm)
target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})

message("-- ${PROJECT_NAME} -- Using Compiler - Path:" ${CMAKE_CXX_COMPILER} "\tVersion:" ${CMAKE_CXX_COMPILER_VERSION} "\tCompiler:" ${CMAKE_CXX_COMPILER_ID}${ColourReset})
message("-- ${PROJECT_NAME} -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
message("-- ${PROJECT_NAME} -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")