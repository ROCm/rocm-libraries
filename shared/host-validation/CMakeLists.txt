# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidation VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOST_VALIDATION_STANDALONE ON)
else()
    set(HOST_VALIDATION_STANDALONE OFF)
endif()

option(HOST_VALIDATION_BUILD_TESTING
       "Build host-validation tests."
       ${HOST_VALIDATION_STANDALONE})
option(HOST_VALIDATION_BUILD_PYTHON
       "Build the host-validation nanobind module."
       OFF)
option(HOST_VALIDATION_BUILD_BLAS_BACKEND
       "Build the optional host-validation CBLAS GEMM backend."
       OFF)

add_library(host-validation-core INTERFACE)
add_library(roc::host-validation-core ALIAS host-validation-core)

add_library(host-validation INTERFACE)
add_library(roc::host-validation ALIAS host-validation)
add_library(host-validation-adapters INTERFACE)
add_library(roc::host-validation-adapters ALIAS host-validation-adapters)

target_compile_features(host-validation-core INTERFACE cxx_std_20)
target_include_directories(host-validation-core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(host-validation-core
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/tensor.hpp"
)

target_compile_features(host-validation INTERFACE cxx_std_20)
target_include_directories(host-validation
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(host-validation
    INTERFACE
        roc::host-validation-core
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
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/detail/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/detail/data_generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/detail/reference_gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/detail/tensor_views.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/validation.hpp"
)

function(roc_host_validation_enable_blas_backend)
    if(TARGET host-validation-blas)
        return()
    endif()

    set(_host_validation_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    find_package(BLAS REQUIRED)

    add_library(host-validation-blas STATIC
        "${_host_validation_source_dir}/src/backends/blas.cpp"
    )
    add_library(roc::host-validation-blas ALIAS host-validation-blas)
    target_compile_features(host-validation-blas PUBLIC cxx_std_20)
    target_link_libraries(host-validation-blas
        PUBLIC
            roc::host-validation
    )
    if(TARGET BLAS::BLAS)
        target_link_libraries(host-validation-blas PRIVATE BLAS::BLAS)
    else()
        target_link_libraries(host-validation-blas PRIVATE ${BLAS_LIBRARIES})
    endif()
    target_sources(host-validation-blas
        PUBLIC
            FILE_SET HEADERS
            BASE_DIRS "${_host_validation_source_dir}/include"
            FILES
                "${_host_validation_source_dir}/include/roc/host_validation/backends/blas.hpp"
    )
    set_target_properties(host-validation-blas
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
    )
    install(TARGETS host-validation-blas
        EXPORT host-validation-targets
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT runtime
        FILE_SET HEADERS
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT devel
    )
endfunction()

if(HOST_VALIDATION_BUILD_BLAS_BACKEND)
    roc_host_validation_enable_blas_backend()
endif()

if(HOST_VALIDATION_BUILD_TESTING)
    enable_testing()
endif()

if(HOST_VALIDATION_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(HOST_VALIDATION_BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(TARGETS host-validation-core host-validation
    EXPORT host-validation-targets
    FILE_SET HEADERS
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
)

set(HOST_VALIDATION_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostValidation"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCHostValidationConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfig.cmake"
    INSTALL_DESTINATION "${HOST_VALIDATION_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT host-validation-targets
    FILE ROCHostValidationTargets.cmake
    DESTINATION "${HOST_VALIDATION_INSTALL_CMAKEDIR}"
    NAMESPACE roc::
    COMPONENT devel
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfigVersion.cmake"
    DESTINATION "${HOST_VALIDATION_INSTALL_CMAKEDIR}"
    COMPONENT devel
)
