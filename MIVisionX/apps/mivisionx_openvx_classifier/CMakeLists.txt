################################################################################
#
# MIT License
#
# Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.
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

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

project(classifier)

# ROCm Path
set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
# avoid setting the default installation path to /usr/local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "MIVisionX default installation path" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add Default libdir
set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Library install directory")
include(GNUInstallDirs)

# OpenCV -- Display Component
find_package(OpenCV REQUIRED)

# Application Includes & Libraries
include_directories(${PROJECT_SOURCE_DIR}/include ${ROCM_PATH}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx)
link_directories(${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR})

# Application Source Files
add_library(caffeModels SHARED source/caffeModels.cpp)
target_link_libraries(caffeModels openvx vx_nn pthread)

add_executable(${PROJECT_NAME} source/classifier.cpp)

# Link Libraries & Set CXX Flags
target_link_libraries(${PROJECT_NAME} openvx vx_nn caffeModels pthread)
# -msse4.2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 built-in functions and code generation.
# -mf16c   -- Support F16C built-in functions and code generation.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c")

# OpenCV 3/4 Support
if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
	message("-- ${PROJECT_NAME} -- OpenCV Version Supported -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X")
	if(${OpenCV_VERSION_MAJOR} EQUAL 4)
		target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=1)
    else()
		target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENCV_4=0)
    endif()
	include_directories(${OpenCV_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
else()
	message(FATAL_ERROR "${PROJECT_NAME} -- OpenCV -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
endif()

# Install Application in ROCm Bin Directory
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
