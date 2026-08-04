# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidation VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOST_VALIDATION_STANDALONE ON)
else()
    set(HOST_VALIDATION_STANDALONE OFF)
endif()

option(HOST_VALIDATION_BUILD_TESTING
       "Build host-validation tests."
       ${HOST_VALIDATION_STANDALONE})

add_library(host-validation INTERFACE)
add_library(roc::host-validation ALIAS host-validation)

add_library(host-validation-adapters INTERFACE)
add_library(roc::host-validation-adapters ALIAS host-validation-adapters)

target_compile_features(host-validation INTERFACE cxx_std_20)
target_include_directories(host-validation
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(host-validation-adapters INTERFACE cxx_std_20)
target_include_directories(host-validation-adapters
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/adapters/include>
)
target_link_libraries(host-validation-adapters
    INTERFACE
        roc::host-validation
)

target_sources(host-validation
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/data_generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/reference_gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/tensor.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/validation.hpp"
)

if(HOST_VALIDATION_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS host-validation
    EXPORT host-validation-targets
    FILE_SET HEADERS
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
)

install(EXPORT host-validation-targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/host-validation"
    NAMESPACE roc::
    COMPONENT devel
)
