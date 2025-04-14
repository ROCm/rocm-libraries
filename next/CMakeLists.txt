# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ########################################################################

if(NOT PROJECT_IS_TOP_LEVEL)
    cmake_minimum_required(VERSION 3.21)
    project(hipblas-common VERSION 1.0.0 LANGUAGES CXX)
    # For most projects, on non-Windows platforms it is advisable to explicitly
    # set CMAKE_INSTALL_PREFIX to a FHS-compliant /opt/... path.
    # set(CMAKE_INSTALL_PREFIX "/opt/rocm/${PROJECT_NAME}")
    # we conditionally set this to a different location in windows
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(hipblas-common INTERFACE)
add_library(roc::hipblas-common ALIAS hipblas-common)

target_sources(hipblas-common 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/include/hipblas-common.h>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hipblas-common/hipblas-common.h>
)
target_include_directories(hipblas-common
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

get_target_property(hipblas_headers_public hipblas-common INTERFACE_SOURCES)
source_group("Header Files\\Public" FILES ${hipblas_headers_public})

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipblas-common-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion
  )
install(
    TARGETS hipblas-common
    EXPORT hipblas-common-targets
    COMPONENT devel
)
install(
    EXPORT hipblas-common-targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-common"
    NAMESPACE roc::
    COMPONENT devel
    FILE hipblas-common-config-targets.cmake
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hipblas-common-config-version.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipblas-common-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-common)
install(
    FILES ${hipblas_headers_public}
    COMPONENT devel
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblas-common"
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
    COMPONENT doc
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)
#export(EXPORT hipblas-common-targets FILE hipblas-common-config.cmake)