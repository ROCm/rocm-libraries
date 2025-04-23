# Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.10)

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()

project(vx_loomio_media)

find_package(OpenCL REQUIRED)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVUTIL_LIBRARY avutil)
find_library(AVCODEC_LIBRARY avcodec)
find_library(SWSCALE_LIBRARY swscale)
find_library(AVFORMAT_LIBRARY avformat)

include_directories(
                    ${OpenCL_INCLUDE_DIRS} 
                    ${OpenCL_INCLUDE_DIRS}/Headers
                    ${AVUTIL_INCLUDE_DIR} 
                    ${AVCODEC_INCLUDE_DIR} 
                    ${SWSCALE_INCLUDE_DIR} 
                    ${AVFORMAT_INCLUDE_DIR}
                    ../../amd_openvx/openvx/include
                    )

list(APPEND SOURCES decoder.cpp encoder.cpp vx_loomio_media.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} ${AVUTIL_LIBRARY} ${AVCODEC_LIBRARY} ${SWSCALE_LIBRARY} ${AVFORMAT_LIBRARY} openvx)
# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -msse4.2")
endif()
