# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")

include(GNUInstallDirs)
include(CMakeDependentOption)
include(fetch_rocm_cmake)
include(hipblaslt_python)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
    endif()
endif()

option(TENSILELITE_ENABLE_HOST "Build the tensilelite host library." ON)
option(TENSILELITE_ENABLE_CLIENT "Build tensilelite client" OFF)
option(TENSILELITE_BUILD_TESTING "Build tensilelite tests" OFF)
option(HIPBLASLT_ENABLE_YAML "Use YAML for parsing configuration files." OFF)
option(HIPBLASLT_ENABLE_THEROCK "Build for TheRock." OFF)
cmake_dependent_option(HIPBLASLT_ENABLE_MXDATAGENERATOR "Use mxDataGenerator for pre-swizzled MX scale strides." ON "NOT WIN32" OFF)

if(TENSILELITE_ENABLE_CLIENT OR TENSILELITE_BUILD_TESTING OR TENSILELITE_ENABLE_AUTOBUILD)
    message(FATAL_ERROR "The tensilelite client, tests and autobuild scripts need the hipBLASLt build; configure projects/hipblaslt instead.")
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../llvm/lib" CACHE STRING "Install RPATH")

find_package(hip REQUIRED)
hipblaslt_find_python(Development.Module)
