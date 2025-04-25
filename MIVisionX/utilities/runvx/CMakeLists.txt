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

project(runvx)

find_package(OpenCV QUIET)

include_directories(../../amd_openvx/openvx/include)

list(APPEND SOURCES
    runvx.cpp
    vxArray.cpp
    vxConvolution.cpp
    vxDistribution.cpp
    vxEngine.cpp
    vxEngineUtil.cpp
    vxImage.cpp
    vxLUT.cpp
    vxMatrix.cpp
    vxParameter.cpp
    vxParamHelper.cpp
    vxPyramid.cpp
    vxRemap.cpp
    vxScalar.cpp
    vxThreshold.cpp
    vxTensor.cpp
    vxUtils.cpp
    )

add_executable(runvx ${SOURCES})
target_link_libraries(runvx openvx)
set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(ENABLE_OPENCL 1)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(runvx ${OpenCL_LIBRARIES})
    else()
        message("-- ${Yellow}NOTE: OpenCL Not Found -- runVX built for CPU only${ColourReset}")
    endif()
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message("-- ${White}runVX -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        set(ENABLE_HIP 1)
        set_target_properties(runvx PROPERTIES LINKER_LANGUAGE CXX)
        set_target_properties(runvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
        include_directories(${ROCM_PATH}/include)
        link_directories(${HIP_PATH}/lib)
        target_link_libraries(runvx hip::host)
        message("-- ${Green}runvx built with HIP Support${ColourReset}")
    else()
        message("-- ${Yellow}NOTE: HIP Not Found -- runVX built for CPU only${ColourReset}")
    endif()
else()
    message("-- ${Yellow}NOTE: GPU Support Not Found or Turned OFF -- runVX built for CPU only${ColourReset}")
endif()

target_compile_definitions(runvx PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(runvx PUBLIC ENABLE_HIP=${ENABLE_HIP})

if(OpenCV_FOUND)
    if(${OpenCV_VERSION_MAJOR} EQUAL 3 OR ${OpenCV_VERSION_MAJOR} EQUAL 4)
        target_compile_definitions(runvx PUBLIC ENABLE_OPENCV=1)
        include_directories(${OpenCV_INCLUDE_DIRS})
        target_link_libraries(runvx ${OpenCV_LIBRARIES})
        if(${OpenCV_VERSION_MAJOR} EQUAL 4)
            target_compile_definitions(runvx PUBLIC USE_OPENCV_4=1)
        else()
            target_compile_definitions(runvx PUBLIC USE_OPENCV_4=0)
        endif()
    else()
        target_compile_definitions(runvx PUBLIC ENABLE_OPENCV=0)
        message("-- ${Yellow}NOTE: RunVX -- OpenCV Version-${OpenCV_VERSION_MAJOR}.${OpenCV_VERSION_MINOR}.X Not Supported${ColourReset}")
    endif()
else(OpenCV_FOUND)
    target_compile_definitions(runvx PUBLIC ENABLE_OPENCV=0)
    message("-- ${Yellow}NOTE: OpenCV Not Found -- runVX built without display support${ColourReset}")
endif(OpenCV_FOUND)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
else()
    target_link_libraries(runvx pthread)
    find_path(SSL_INCLUDE
            NAMES openssl/hmac.h openssl/md5.h
            HINTS /usr/local/opt/openssl/include /usr/include /usr/local/include /opt/local/include
    )
    if(NOT "${SSL_INCLUDE}" STREQUAL "SSL_INCLUDE-NOTFOUND")
        include_directories(${SSL_INCLUDE})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_OpenSSL=1")
        target_link_libraries(runvx crypto)
    endif()
    # -Wno-deprecated-declarations -- Do not warn about uses of functions, variables, and types marked as deprecated by using the deprecated attribute.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

# install MIVisionX executables -- {ROCM_PATH}/bin
install(TARGETS runvx DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)

# install MIVisionX apps & samples -- {ROCM_PATH}/share/mivisionx/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/gdf DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)

message("-- ${White}runVX -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
