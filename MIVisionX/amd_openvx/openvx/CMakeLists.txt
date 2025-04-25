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
set(VERSION "1.3.0")

project(openvx VERSION ${VERSION} LANGUAGES CXX)

include_directories(include ago api)

list(APPEND SOURCES
    ago/ago_drama.cpp
    ago/ago_drama_alloc.cpp
    ago/ago_drama_analyze.cpp
    ago/ago_drama_divide.cpp
    ago/ago_drama_merge.cpp
    ago/ago_drama_remove.cpp
    ago/ago_haf_cpu.cpp
    ago/ago_haf_cpu_arithmetic.cpp
    ago/ago_haf_cpu_canny.cpp
    ago/ago_haf_cpu_ch_extract_combine.cpp
    ago/ago_haf_cpu_color_convert.cpp
    ago/ago_haf_cpu_fast_corners.cpp
    ago/ago_haf_cpu_filter.cpp
    ago/ago_haf_cpu_geometric.cpp
    ago/ago_haf_cpu_generic_functions.cpp
    ago/ago_haf_cpu_harris.cpp
    ago/ago_haf_cpu_histogram.cpp
    ago/ago_haf_cpu_logical.cpp
    ago/ago_haf_cpu_opticalflow.cpp
    ago/ago_haf_cpu_pyramid.cpp
    ago/ago_haf_gpu_common.cpp
    ago/ago_haf_gpu_conversion.cpp
    ago/ago_haf_gpu_corners.cpp
    ago/ago_haf_gpu_linear_filter.cpp
    ago/ago_haf_gpu_special_filters.cpp
    ago/ago_interface.cpp
    ago/ago_kernel_api.cpp
    ago/ago_kernel_list.cpp
    ago/ago_platform.cpp
    ago/ago_util.cpp
    ago/ago_util_opencl.cpp
    ago/ago_util_hip.cpp
    api/vxu.cpp
    api/vx_api.cpp
    api/vx_nodes.cpp
)

add_library(openvx SHARED ${SOURCES})
add_library(vxu SHARED api/vxu.cpp ago/ago_platform.cpp)
set(ENABLE_OPENCL 0)
set(ENABLE_HIP 0)

# Backend Specific Settings
if (GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        set(ENABLE_OPENCL 1)
        include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
        target_link_libraries(openvx ${OpenCL_LIBRARIES})
        message("-- ${Green}AMD OpenVX -- OpenVX built with OpenCL Support${ColourReset}")
        message("-- ${Blue}Using OpenCL Library -- ${OpenCL_LIBRARIES}${ColourReset}")
    else()
        set(GPU_SUPPORT OFF)
        message("-- ${Red}WARNING: OpenCL Not Found -- OpenVX built for CPU only${ColourReset}")
    endif()
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    SET(OpenCL_FOUND FALSE)
    if(HIP_FOUND)
        message("-- ${White}AMD OpenVX -- Using HIP -- Path:" ${HIP_PATH} "\tVersion:" ${HIP_VERSION} "\tCompiler:" ${HIP_COMPILER} ${ColourReset})
        set(ENABLE_HIP 1)
        set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
        set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
        include_directories(${ROCM_PATH}/include)
        include_directories(hipvx)
        link_directories(${HIP_PATH}/lib)
        target_link_libraries(openvx hip::host)
        add_dependencies(openvx openvx_hip)
        target_link_libraries(openvx $<TARGET_OBJECTS:openvx_hip>)
        message("-- ${Green}AMD OpenVX -- OpenVX built with HIP Support${ColourReset}")
    else()
        set(GPU_SUPPORT OFF)
        message("-- ${Red}WARNING: HIP Not Found -- OpenVX built for CPU only${ColourReset}")
    endif()
else()
    message("-- ${Red}WARNING: OpenCL/HIP Not Found or GPU Support Turned OFF -- OpenVX built for CPU only${ColourReset}")
endif()

target_compile_definitions(openvx PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(openvx PUBLIC ENABLE_HIP=${ENABLE_HIP})
target_compile_definitions(vxu PUBLIC ENABLE_OPENCL=${ENABLE_OPENCL})
target_compile_definitions(vxu PUBLIC ENABLE_HIP=${ENABLE_HIP})
target_link_libraries(vxu openvx)

set_target_properties(openvx PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(vxu PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS openvx LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS openvx LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS openvx LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
install(TARGETS vxu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS vxu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS vxu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

# install MIVisionX backend type files -- {ROCM_PATH}/include/mivisionx/
configure_file("${PROJECT_SOURCE_DIR}/include/openvx_backend.h.in" "${PROJECT_BINARY_DIR}/include/mivisionx/openvx_backend.h")
install(FILES ${PROJECT_BINARY_DIR}/include/mivisionx/openvx_backend.h DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)

# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_ext_amd.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
install(FILES
            include/VX/vx.h
            include/VX/vx_api.h
            include/VX/vx_compatibility.h
            include/VX/vx_import.h
            include/VX/vx_kernels.h
            include/VX/vx_khr_buffer_aliasing.h
            include/VX/vx_khr_class.h
            include/VX/vx_khr_icd.h
            include/VX/vx_khr_import_kernel.h
            include/VX/vx_khr_ix.h
            include/VX/vx_khr_nn.h
            include/VX/vx_khr_opencl_interop.h
            include/VX/vx_khr_pipelining.h
            include/VX/vx_khr_tiling.h
            include/VX/vx_khr_user_data_object.h
            include/VX/vx_khr_xml.h
            include/VX/vx_nodes.h
            include/VX/vx_types.h
            include/VX/vx_vendors.h
            include/VX/vxu.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx/VX COMPONENT dev)
# install MIVisionX apps & samples -- {ROCM_PATH}/share/mivisionx/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/bubble_pop DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/optical_flow DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/c_samples DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/images DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /DVX_API_ENTRY=__declspec(dllexport)")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /DVX_API_ENTRY=__declspec(dllexport)")
else()
    # -msse4.2 -- Support MMX, SSE, SSE2, SSE3, SSSE3, SSE4.1 and SSE4.2 built-in functions and code generation.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    target_link_libraries(openvx dl m)
endif()

message("-- ${White}AMD OpenVX -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
