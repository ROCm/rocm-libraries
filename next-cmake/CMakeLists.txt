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

cmake_minimum_required(VERSION 3.11)
project(hipblas-common VERSION 1.0.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(hipblas-common INTERFACE)
add_library(hip::hipblas-common ALIAS hipblas-common)
add_library(roc::hipblas-common ALIAS hipblas-common)
message(DEPRECATION "roc::hipblas-common is deprecated. Use hip::hipblas-common")

target_sources(hipblas-common 
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../library/include/hipblas-common/hipblas-common.h>
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

# Installing two target files one with hip:: namespace and one with
# roc:: namespace which is on deprecation path
install(
    TARGETS hipblas-common
    EXPORT roc-hipblas-common-targets
    COMPONENT hipBLAS-common_Development
)
install(
    EXPORT roc-hipblas-common-targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-common"
    NAMESPACE roc::
    COMPONENT hipBLAS-common_Development
    FILE roc-hipblas-common-config-targets.cmake
)
install(
    TARGETS hipblas-common
    EXPORT hipblas-common-targets
    COMPONENT hipBLAS-common_Development
)
install(
    EXPORT hipblas-common-targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-common"
    NAMESPACE hip::
    COMPONENT hipBLAS-common_Development
    FILE hipblas-common-config-targets.cmake
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/hipblas-common-config-version.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/hipblas-common-config.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-common)
install(
    FILES ${hipblas_headers_public}
    COMPONENT hipBLAS-common_Development
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/hipblas-common"
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md"
    COMPONENT hipBLAS-common_Docs
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
)
export(EXPORT hipblas-common-targets FILE hipblas-common-config.cmake)

set(CPACK_PACKAGE_NAME hipBLAS-common)
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_PACKAGE_CONTACT "hipBLAS Maintainer <hipblas-maintainer@amd.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Common files shared by hipBLAS and hipBLASLt")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_VERBATIM_VARIABLES YES)
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_LIST_DIR}/../LICENSE.md)
set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_LIST_DIR}/../README.md)
set(CPACK_COMPONENTS_ALL "hipBLAS-common_Development;hipBLAS-common_Docs")

set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_PROVIDES "hipblas-common (= 1.0.0)")
set(CPACK_DEB_COMPONENT_INSTALL "ON")

set(CPACK_DEBIAN_HIPBLAS-COMMON_DEVELOPMENT_PACKAGE_NAME "hipblas-common-dev")
set(CPACK_DEBIAN_HIPBLAS-COMMON_DOCS_PACKAGE_NAME "hipblas-common-doc")
set(CPACK_DEBIAN_HIPBLAS-COMMON_DOCS_PACKAGE_SECTION "doc")
#set(CPACK_DEBIAN_PACKAGE_RELEASE "some git hash") # there is come etra functionality in rocm-cmake to compute this hash why not use the command line?
#set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core") # I don't think we really need this to build

set(CPACK_RPM_COMPONENT_INSTALL "ON")
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
set(CPACK_RPM_PACKAGE_AUTOREQPROV "Off")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
#set(CPACK_RPM_PACKAGE_RELOCATABLE "Off") why?
#set(CPACK_RPM_PACKAGE_AUTOREQPROV "Off") why?
#set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
#set(CPACK_RPM_PACKAGE_RELEASE "hash")

set(CPACK_RPM_HIPBLAS-COMMON_DEVELOPMENT_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_HIPBLAS-COMMON_DEVELOPMENT_PACKAGE_NAME "hipblas-common-devel")
set(CPACK_RPM_HIPBLAS-COMMON_DEVELOPMENT_PACKAGE_PROVIDES "hipblas-common = 1.0.0")

set(CPACK_RPM_HIPBLAS-COMMON_DOCS_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_HIPBLAS-COMMON_DOCS_PACKAGE_NAME "hipblas-common-doc")

include(CPack)

cpack_add_component(hipBLAS-common_Development
    DISPLAY_NAME "Developer pre-requisites"
    DESCRIPTION "hipBLAS-common header files required for development"
)

cpack_add_component(hipBLAS-common_Docs
    DISPLAY_NAME "License file"
    DISABLED
)
