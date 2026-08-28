# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostNumerics VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(HOST_NUMERICS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostNumerics"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOST_NUMERICS_STANDALONE ON)
else()
    set(HOST_NUMERICS_STANDALONE OFF)
endif()

option(HOST_NUMERICS_BUILD_TESTING
       "Build host-numerics tests."
       ${HOST_NUMERICS_STANDALONE})
option(HOST_NUMERICS_BUILD_PYTHON
       "Build the host-numerics nanobind module."
       ${HOST_NUMERICS_STANDALONE})
option(HOST_NUMERICS_BUILD_BLAS_BACKEND
       "Build the optional host-numerics CBLAS GEMM backend."
       OFF)
option(HOST_NUMERICS_BUILD_BENCHMARKS
       "Build host-numerics development benchmarks."
       OFF)
option(HOST_NUMERICS_ENABLE_FETCH
       "Allow host-numerics tests to fetch missing dependencies."
       ${HOST_NUMERICS_STANDALONE})
option(HOST_NUMERICS_AMD_GPU_LAYOUT_ENABLE_OPENMP
       "Use OpenMP for large AMD GPU physical-layout transforms when available."
       ON)

# Install one public target and its component-specific export file.
function(
    roc_host_numerics_install_component_target
    target_name
    export_name
    export_file
)
    # cmake-lint: disable=E1122
    install(TARGETS "${target_name}"
        EXPORT "${export_name}"
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT devel
        FILE_SET HEADERS
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            COMPONENT devel
    )
    install(EXPORT "${export_name}"
        FILE "${export_file}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostNumerics"
        NAMESPACE roc::
        COMPONENT devel
    )
endfunction()

add_library(host-numerics-core INTERFACE)
add_library(roc::host-numerics-core ALIAS host-numerics-core)

add_library(host-numerics-amd-gpu-layout STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/mx.cpp"
)
add_library(
    roc::host-numerics-amd-gpu-layout
    ALIAS host-numerics-amd-gpu-layout
)

add_library(host-numerics STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/axpby.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/epilogue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm_validation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/layer_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/structured_sparsity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/blocked.cpp"
)
add_library(roc::host-numerics ALIAS host-numerics)

target_compile_features(host-numerics-core INTERFACE cxx_std_20)
target_include_directories(host-numerics-core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(host-numerics-core
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/index_order.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/scalar.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/scalar_codec.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/tensor.hpp"
)

target_compile_features(host-numerics-amd-gpu-layout PUBLIC cxx_std_17)
target_include_directories(host-numerics-amd-gpu-layout
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(host-numerics-amd-gpu-layout
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/amd_gpu_layout/mx.hpp"
)
set_target_properties(host-numerics-amd-gpu-layout
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VERIFY_INTERFACE_HEADER_SETS ON
        VISIBILITY_INLINES_HIDDEN ON
)
if(HOST_NUMERICS_AMD_GPU_LAYOUT_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            host-numerics-amd-gpu-layout
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        set_property(
            TARGET host-numerics-amd-gpu-layout
            PROPERTY ROC_HOST_NUMERICS_USES_OPENMP ON
        )
    endif()
endif()

target_compile_features(host-numerics PUBLIC cxx_std_20)
target_include_directories(host-numerics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(host-numerics
    PUBLIC
        roc::host-numerics-core
)
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(host-numerics PRIVATE OpenMP::OpenMP_CXX)
    set_property(
        TARGET host-numerics
        PROPERTY ROC_HOST_NUMERICS_USES_OPENMP ON
    )
endif()

target_sources(host-numerics
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/axpby.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/epilogue.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/gemm_validation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/layer_norm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/mx.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/operation_types.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/reduction.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/softmax.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/structured_sparsity.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_numerics/validation.hpp"
)
set_target_properties(host-numerics
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

# Enables the optional BLAS-backed GEMM implementation. This function is
# intentionally callable by an enclosing source build after the base target
# has already been added.
function(roc_host_numerics_enable_blas_backend)
    if(TARGET host-numerics-blas)
        return()
    endif()

    set(_host_numerics_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    list(PREPEND CMAKE_MODULE_PATH "${_host_numerics_source_dir}/cmake")
    find_package(CBLAS REQUIRED MODULE)

    add_library(host-numerics-blas STATIC
        "${_host_numerics_source_dir}/src/backends/blas.cpp"
    )
    add_library(roc::host-numerics-blas ALIAS host-numerics-blas)
    target_compile_features(host-numerics-blas PUBLIC cxx_std_20)
    target_include_directories(host-numerics-blas
        PRIVATE
            "${_host_numerics_source_dir}/src"
    )
    target_link_libraries(host-numerics-blas
        PUBLIC
            roc::host-numerics
        PRIVATE
            CBLAS::CBLAS
    )
    target_sources(host-numerics-blas
        PUBLIC
            FILE_SET HEADERS
            BASE_DIRS "${_host_numerics_source_dir}/include"
            FILES
                "${_host_numerics_source_dir}/include/roc/host_numerics/backends/blas.hpp"
    )
    set_target_properties(host-numerics-blas
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )
    roc_host_numerics_install_component_target(
        host-numerics-blas
        host-numerics-blas-targets
        ROCHostNumericsBLASTargets.cmake
    )
endfunction()

if(HOST_NUMERICS_BUILD_BLAS_BACKEND)
    roc_host_numerics_enable_blas_backend()
endif()

if(HOST_NUMERICS_BUILD_TESTING)
    enable_testing()
endif()

if(HOST_NUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(HOST_NUMERICS_BUILD_TESTING)
    add_subdirectory(tests)
    add_test(
        NAME host-numerics-package-components
        COMMAND
            "${CMAKE_COMMAND}"
            "-DCOMPONENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DCOMPONENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DPACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package"
            "-DPACKAGE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/tests/package/component-package-test"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DTEST_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DTEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DTEST_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DTEST_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DTEST_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
            "-DTEST_CONFIG=$<CONFIG>"
            "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package/run_package_component_tests.cmake"
    )
    set_tests_properties(
        host-numerics-package-components
        PROPERTIES
            RUN_SERIAL TRUE
    )
endif()

roc_host_numerics_install_component_target(
    host-numerics-core
    host-numerics-core-targets
    ROCHostNumericsCoreTargets.cmake
)
roc_host_numerics_install_component_target(
    host-numerics
    host-numerics-operations-targets
    ROCHostNumericsOperationsTargets.cmake
)
roc_host_numerics_install_component_target(
    host-numerics-amd-gpu-layout
    host-numerics-amd-gpu-layout-targets
    ROCHostNumericsAMDGPULayoutTargets.cmake
)

set_property(
    GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_INSTALL_CMAKEDIR
    "${HOST_NUMERICS_INSTALL_CMAKEDIR}"
)
set_property(
    GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_VERSION
    "${PROJECT_VERSION}"
)

# Verify that generic numerical targets do not depend on the AMD-specific leaf.
function(roc_host_numerics_assert_generic_target_boundary target_name)
    if(NOT TARGET "${target_name}")
        return()
    endif()
    foreach(property_name LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(link_libraries "${target_name}" "${property_name}")
        if(link_libraries
           AND link_libraries MATCHES "host-numerics-amd-gpu-layout")
            message(FATAL_ERROR
                "Generic host-numerics target ${target_name} depends on "
                "the AMD GPU layout module through ${property_name}."
            )
        endif()
    endforeach()
endfunction()

# Finalize package metadata after parent projects may enable optional backends.
function(roc_host_numerics_finalize_package)
    get_property(
        _host_numerics_source_dir
        GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_SOURCE_DIR
    )
    get_property(
        _host_numerics_binary_dir
        GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_BINARY_DIR
    )
    get_property(
        _host_numerics_install_cmakedir
        GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_INSTALL_CMAKEDIR
    )
    get_property(
        _host_numerics_version
        GLOBAL PROPERTY ROC_HOST_NUMERICS_PACKAGE_VERSION
    )

    set(HOST_NUMERICS_PACKAGE_HAS_BLAS OFF)
    if(TARGET host-numerics-blas)
        set(HOST_NUMERICS_PACKAGE_HAS_BLAS ON)
    endif()
    set(HOST_NUMERICS_PACKAGE_OPERATIONS_HAS_OPENMP OFF)
    get_target_property(
        _host_numerics_uses_openmp
        host-numerics
        ROC_HOST_NUMERICS_USES_OPENMP
    )
    if(_host_numerics_uses_openmp)
        set(HOST_NUMERICS_PACKAGE_OPERATIONS_HAS_OPENMP ON)
    endif()

    set(HOST_NUMERICS_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP OFF)
    get_target_property(
        _host_numerics_amd_gpu_layout_uses_openmp
        host-numerics-amd-gpu-layout
        ROC_HOST_NUMERICS_USES_OPENMP
    )
    if(_host_numerics_amd_gpu_layout_uses_openmp)
        set(HOST_NUMERICS_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP ON)
    endif()

    foreach(generic_target
            host-numerics-core
            host-numerics
            host-numerics-blas)
        roc_host_numerics_assert_generic_target_boundary("${generic_target}")
    endforeach()

    configure_package_config_file(
        "${_host_numerics_source_dir}/cmake/ROCHostNumericsConfig.cmake.in"
        "${_host_numerics_binary_dir}/ROCHostNumericsConfig.cmake"
        INSTALL_DESTINATION "${_host_numerics_install_cmakedir}"
    )
    write_basic_package_version_file(
        "${_host_numerics_binary_dir}/ROCHostNumericsConfigVersion.cmake"
        VERSION "${_host_numerics_version}"
        COMPATIBILITY SameMajorVersion
    )
endfunction()

cmake_language(
    DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL roc_host_numerics_finalize_package
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostNumericsConfigVersion.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCBLAS.cmake"
    DESTINATION "${HOST_NUMERICS_INSTALL_CMAKEDIR}"
    COMPONENT devel
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/ROCHostNumerics"
    COMPONENT docs
)
