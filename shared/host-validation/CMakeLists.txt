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
       ${HOST_VALIDATION_STANDALONE})
option(HOST_VALIDATION_BUILD_BLAS_BACKEND
       "Build the optional host-validation CBLAS GEMM backend."
       OFF)
option(HOST_VALIDATION_BUILD_MX_BACKEND
       "Build the optional host-validation MX data-generation backend."
       ${HOST_VALIDATION_STANDALONE})
option(HOST_VALIDATION_BUILD_BENCHMARKS
       "Build host-validation development benchmarks."
       OFF)

add_library(host-validation-core INTERFACE)
add_library(roc::host-validation-core ALIAS host-validation-core)

add_library(host-validation STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/axpby.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/epilogue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/layer_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/structured_sparsity.cpp"
)
add_library(roc::host-validation ALIAS host-validation)

add_library(host-validation-tiled STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/tiled.cpp"
)
add_library(roc::host-validation-tiled ALIAS host-validation-tiled)

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

target_compile_features(host-validation PUBLIC cxx_std_20)
target_include_directories(host-validation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(host-validation
    PUBLIC
        roc::host-validation-core
)

target_sources(host-validation
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/axpby.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/epilogue.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/generation_primitives.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/layer_norm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/operation_types.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/reduction.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/softmax.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/structured_sparsity.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/typed_comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/validation.hpp"
)
set_target_properties(host-validation
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

target_compile_features(host-validation-tiled PUBLIC cxx_std_20)
target_link_libraries(host-validation-tiled PUBLIC roc::host-validation)
target_include_directories(host-validation-tiled
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_sources(host-validation-tiled
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/backends/tiled.hpp"
)
set_target_properties(host-validation-tiled
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

# Add the optional BLAS-backed GEMM implementation to the current build.
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
    target_include_directories(host-validation-blas
        PRIVATE
            "${_host_validation_source_dir}/src"
    )
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
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )
    # cmake-lint: disable=E1122
    install(TARGETS host-validation-blas
        EXPORT host-validation-targets
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT devel
        FILE_SET HEADERS
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT devel
    )
endfunction()

if(HOST_VALIDATION_BUILD_BLAS_BACKEND)
    roc_host_validation_enable_blas_backend()
endif()

# Add the optional MX generation implementation to the current build.
function(roc_host_validation_enable_mx_backend)
    if(TARGET host-validation-mx)
        return()
    endif()

    set(_host_validation_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    find_package(OpenMP QUIET COMPONENTS CXX)
    add_library(host-validation-mx STATIC
        "${_host_validation_source_dir}/src/mx.cpp"
    )
    add_library(roc::host-validation-mx ALIAS host-validation-mx)
    target_compile_features(host-validation-mx PUBLIC cxx_std_20)
    target_link_libraries(host-validation-mx
        PUBLIC
            roc::host-validation-core
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(host-validation-mx PRIVATE OpenMP::OpenMP_CXX)
        set_property(
            TARGET host-validation-mx
            PROPERTY ROC_HOST_VALIDATION_USES_OPENMP ON
        )
    endif()
    target_sources(host-validation-mx
        PUBLIC
            FILE_SET HEADERS
            BASE_DIRS "${_host_validation_source_dir}/include"
            FILES
                "${_host_validation_source_dir}/include/roc/host_validation/mx.hpp"
    )
    set_target_properties(host-validation-mx
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )
    # cmake-lint: disable=E1122
    install(TARGETS host-validation-mx
        EXPORT host-validation-targets
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT devel
        FILE_SET HEADERS
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT devel
    )
endfunction()

if(HOST_VALIDATION_BUILD_MX_BACKEND)
    roc_host_validation_enable_mx_backend()
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

# cmake-lint: disable=E1122
install(TARGETS host-validation-core host-validation host-validation-tiled
    EXPORT host-validation-targets
    ARCHIVE
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT devel
    FILE_SET HEADERS
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT devel
)

set(HOST_VALIDATION_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostValidation"
)

set_property(
    GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_INSTALL_CMAKEDIR
    "${HOST_VALIDATION_INSTALL_CMAKEDIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_VERSION
    "${PROJECT_VERSION}"
)

# Finalize package metadata after parent projects may enable optional backends.
function(roc_host_validation_finalize_package)
    get_property(
        _host_validation_source_dir
        GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_SOURCE_DIR
    )
    get_property(
        _host_validation_binary_dir
        GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_BINARY_DIR
    )
    get_property(
        _host_validation_install_cmakedir
        GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_INSTALL_CMAKEDIR
    )
    get_property(
        _host_validation_version
        GLOBAL PROPERTY ROC_HOST_VALIDATION_PACKAGE_VERSION
    )

    set(HOST_VALIDATION_PACKAGE_HAS_BLAS OFF)
    if(TARGET host-validation-blas)
        set(HOST_VALIDATION_PACKAGE_HAS_BLAS ON)
    endif()
    set(HOST_VALIDATION_PACKAGE_HAS_OPENMP OFF)
    if(TARGET host-validation-mx)
        get_target_property(
            _host_validation_mx_uses_openmp
            host-validation-mx
            ROC_HOST_VALIDATION_USES_OPENMP
        )
        if(_host_validation_mx_uses_openmp)
            set(HOST_VALIDATION_PACKAGE_HAS_OPENMP ON)
        endif()
    endif()

    configure_package_config_file(
        "${_host_validation_source_dir}/cmake/ROCHostValidationConfig.cmake.in"
        "${_host_validation_binary_dir}/ROCHostValidationConfig.cmake"
        INSTALL_DESTINATION "${_host_validation_install_cmakedir}"
    )
    write_basic_package_version_file(
        "${_host_validation_binary_dir}/ROCHostValidationConfigVersion.cmake"
        VERSION "${_host_validation_version}"
        COMPATIBILITY SameMajorVersion
    )
endfunction()

cmake_language(
    DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL roc_host_validation_finalize_package
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
