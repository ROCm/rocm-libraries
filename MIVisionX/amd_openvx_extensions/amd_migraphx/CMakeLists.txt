# Copyright (c) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
set(VERSION "1.0.0")

project(vx_amd_migraphx VERSION ${VERSION} LANGUAGES CXX)

list (APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)

# TBD: HIP required to build vx_migraphx
if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    if(NOT DEFINED HIP_PATH)
        if(NOT DEFINED ENV{HIP_PATH})
            set(HIP_PATH ${ROCM_PATH})
        else()
            set(HIP_PATH $ENV{HIP_PATH})
        endif()
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP QUIET)

    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)
    # TBD: migraphx package required to build vx_migraphx
    find_package(migraphx QUIET)

    include_directories(../../amd_openvx/openvx/include
    ${ROCM_PATH}/include
    include)

    list(APPEND SOURCES
            src/kernels.cpp
            src/migraphx_node.cpp
        )

    if (HIP_FOUND AND migraphx_FOUND)
        message("-- ${Green}${PROJECT_NAME} -- Building with HIP backend${ColourReset}")
        set(ENABLE_HIP 1)
        add_definitions(-DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
        link_directories(${HIP_PATH}/lib)
        add_library(${PROJECT_NAME} SHARED ${SOURCES})
        target_link_libraries(${PROJECT_NAME} openvx migraphx::c hip::host)
        set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
        # install MIVisionX libs -- {ROCM_PATH}/lib
        install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
        install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
        install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
        # install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
        install(FILES include/vx_amd_migraphx.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
        # install - test
        install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/amd_migraphx_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)
    else()
        message("-- ${Red}WARNING: MIGraphX or HIP Not Found -- ${PROJECT_NAME} module excluded${ColourReset}")
    endif()

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else()
        # -msse4.2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 built-in functions and code generation.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    endif()
endif()

message("-- ${White}vx_migraphx -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
