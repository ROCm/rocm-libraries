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

project (mvobjdetect)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenCV QUIET)

set(ROCM_PATH /opt/rocm CACHE PATH "ROCm Installation Path")
#find the OPENVX backend type - TBD: ADD FindMIVisionX.cmake
set(OPENVX_BACKEND_OPENCL_FOUND 0)
set(OPENVX_BACKEND_HIP_FOUND 0)
if(EXISTS ${ROCM_PATH}/include/mivisionx/openvx_backend.h)
    file(READ ${ROCM_PATH}/include/mivisionx/openvx_backend.h OPENVX_BACKEND_FILE)
    string(REGEX MATCH "ENABLE_OPENCL ([0-9]*)" _ ${OPENVX_BACKEND_FILE})
    set(OPENVX_BACKEND_OPENCL_FOUND ${CMAKE_MATCH_1})
    string(REGEX MATCH "ENABLE_HIP ([0-9]*)" _ ${OPENVX_BACKEND_FILE})
    set(OPENVX_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
else()
    message("-- WARNING: ${ROCM_PATH}/include/mivisionx/openvx_backend.h file Not Found. please install the latest mivisionx!")
endif()

if (OPENVX_BACKEND_OPENCL_FOUND)
    find_package(OpenCL REQUIRED)
    include_directories (${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers )
endif()

include_directories (${ROCM_PATH}/include ${ROCM_PATH}/include/mivisionx ${PROJECT_SOURCE_DIR} )
link_directories    (${ROCM_PATH}/lib ${PROJECT_SOURCE_DIR}/lib)
option (USE_POSTPROC  "Use postprocessing module implementation" ON)
set(SOURCES mvobjdetect.cpp mvdeploy_api.cpp visualize.cpp)
add_executable(mvobjdetect ${SOURCES})

if(OpenCV_FOUND)
     if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
         message("-- OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Supported")
         target_compile_definitions(mvobjdetect PUBLIC ENABLE_OPENCV=1)
         include_directories(${OpenCV_INCLUDE_DIRS})
         target_link_libraries(mvobjdetect ${OpenCV_LIBRARIES})
         if(${OpenCV_VERSION_MAJOR} EQUAL 4)
 	        target_compile_definitions(mvobjdetect PUBLIC USE_OPENCV_4=1)
         else()
 	        target_compile_definitions(mvobjdetect PUBLIC USE_OPENCV_4=0)
         endif()
     else()
         message("-- WARNING: OpenCV Found -- Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported")
         target_compile_definitions(mvobjdetect PUBLIC ENABLE_OPENCV=0)
     endif()
 else()
     message(FATAL_ERROR "OpenCV Not Found -- Display Failure")
     target_compile_definitions(mvobjdetect PUBLIC ENABLE_OPENCV=0)
 endif()

#add optional postprocess module
if (USE_POSTPROC)
  include_directories (mv_extras)
  add_subdirectory (mv_extras)
  set(EXTRA_LIBS ${EXTRA_LIBS} mv_extras)
endif (USE_POSTPROC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mf16c")
target_compile_definitions (mvobjdetect PRIVATE ENABLE_MVDEPLOY=1)
target_link_libraries(mvobjdetect openvx vx_nn vx_amd_media pthread mv_deploy ${EXTRA_LIBS} ${CMAKE_DL_LIBS})
