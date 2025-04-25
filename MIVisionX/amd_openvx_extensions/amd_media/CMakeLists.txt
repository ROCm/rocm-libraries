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
set(VERSION "1.1.0")

project(vx_amd_media VERSION ${VERSION} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    set(ENABLE_OPENCL 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP})
    message("-- ${Green}amd_media -- Building with OpenCL${ColourReset}")
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    set(OpenCL_FOUND FALSE)
    set(ENABLE_HIP 1)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP REQUIRED)
    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${ROCM_PATH}/lib)
    message("-- ${Green}amd_media -- Building with HIP${ColourReset}")
endif()

find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale)

include_directories(${AVUTIL_INCLUDE_DIR}
                    ${AVCODEC_INCLUDE_DIR}
                    ${SWSCALE_INCLUDE_DIR}
                    ${AVFORMAT_INCLUDE_DIR}
                    ../../amd_openvx/openvx/include
                    include
                    )

list(APPEND SOURCES decoder.cpp encoder.cpp kernels.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${AVUTIL_LIBRARY} ${AVCODEC_LIBRARY} ${SWSCALE_LIBRARY} ${AVFORMAT_LIBRARY} openvx)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    target_link_libraries(${PROJECT_NAME} hip::host)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
if(FFMPEG_VERSION_4)
    target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_FFMPEG_VERSION_4=1)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC WITH_FFMPEG_VERSION_4=0)
endif()

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_amd_media.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
# install - test
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/amd_media_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute.
    # -msse4.2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 built-in functions and code generation. 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -msse4.2")
endif()

message("-- ${White}amd_media -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
