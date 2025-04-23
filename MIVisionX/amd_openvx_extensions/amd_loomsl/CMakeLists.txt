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

set(VERSION "1.0.0")

project(amd_loomsl VERSION ${VERSION} LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS} 
		${OpenCL_INCLUDE_DIRS}/Headers 
		../../amd_openvx/openvx/include
)

list(APPEND SOURCES
	kernels/alpha_blend.cpp
	kernels/chroma_key.cpp
	kernels/color_convert.cpp
	kernels/exp_comp.cpp
	kernels/exposure_compensation.cpp
	kernels/kernels.cpp
	kernels/lens_distortion_remap.cpp
	kernels/merge.cpp
	kernels/multiband_blender.cpp
	kernels/noise_filter.cpp
	kernels/pyramid_scale.cpp
	kernels/seam_find.cpp
	kernels/warp.cpp
	kernels/warp_eqr_to_aze.cpp
	kernels/initialize_setup_tables.cpp
	live_stitch_api.cpp
	profiler.cpp
	)

include_directories(. kernels)
add_library(vx_loomsl SHARED ${SOURCES})
target_link_libraries(vx_loomsl ${OpenCL_LIBRARIES} openvx)
set_target_properties(vx_loomsl PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS vx_loomsl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS vx_loomsl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS vx_loomsl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES live_stitch_api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
# install MIVisionX apps & samples -- {ROCM_PATH}/share/mivisionx/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/loom_360_stitch DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)

if(WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

message("-- ${White}amd_loom -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
