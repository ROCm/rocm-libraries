# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Use the monorepo CMake baseline.
# CMake 3.25.2 was released on 2023-01-19.
cmake_minimum_required(VERSION 3.25.2)
project(HostNumerics VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/roc/hostnumerics/version.hpp"
    @ONLY
)

set(HOSTNUMERICS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(HOSTNUMERICS_STANDALONE ON)
else()
    set(HOSTNUMERICS_STANDALONE OFF)
endif()

option(HOSTNUMERICS_BUILD_TESTING
       "Build the standalone C++ and installed-package validation suites."
       ${HOSTNUMERICS_STANDALONE})
option(HOSTNUMERICS_BUILD_PYTHON
       "Build the Python module that exposes tensors and numerical operations through nanobind."
       ${HOSTNUMERICS_STANDALONE})
option(HOSTNUMERICS_BUILD_BLAS_BACKEND
       "Build the optional CBLAS GEMM backend; the provider must use 32-bit (LP64) dimensions."
       OFF)
option(HOSTNUMERICS_BUILD_BENCHMARKS
       "Build standalone development benchmarks; these are not installed."
       OFF)
option(HOSTNUMERICS_ENABLE_FETCH
       "Allow standalone validation builds to fetch missing test dependencies."
       ${HOSTNUMERICS_STANDALONE})
option(HOSTNUMERICS_AMD_GPU_LAYOUT_ENABLE_OPENMP
       "Parallelize large CPU-side AMD GPU layout transforms with OpenMP when available."
       ON)

# Install one public target and its component-specific export file.
function(
    hostnumerics_install_component_target
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
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/HostNumerics"
        NAMESPACE roc::
        COMPONENT devel
    )
endfunction()

# Scalar types and conversions, plus the tensor model (shape, layout, and
# storage), are available independently of numerical operations.
add_library(hostnumerics-core STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_transformations.cpp"
)
add_library(roc::hostnumerics-core ALIAS hostnumerics-core)

add_library(hostnumerics-amd-gpu-layout STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/gfx1250_mx_layout.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/gfx950_mx_layout.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/mx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/mx_layout_common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/pre_swizzle.cpp"
)
add_library(
    roc::hostnumerics-amd-gpu-layout
    ALIAS hostnumerics-amd-gpu-layout
)

# Generation, reference operations, and comparison link core transitively.
# Consumers needing only the tensor/scalar model can link hostnumerics-core.
add_library(hostnumerics STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_operations.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison_accumulator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison_common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison_iteration.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/comparison_metrics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/epilogue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generation_recipe_access.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/generation_values.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/layer_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mx.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mx_generation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mx_packing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reference_validation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/reduction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/structured_sparsity.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/structured_sparsity_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/two_of_four_metadata.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/blocked.cpp"
)
add_library(roc::hostnumerics ALIAS hostnumerics)

target_compile_features(hostnumerics-core PUBLIC cxx_std_20)
target_include_directories(hostnumerics-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(hostnumerics-core
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_BINARY_DIR}/include"
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/include/roc/hostnumerics/version.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/index_order.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/layout.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar_binary_float.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar_codec.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar_conversion.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar_storage.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/scalar_type.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/shape.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/tensor.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/tensor_transformations.hpp"
)
set_target_properties(hostnumerics-core
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

target_compile_features(hostnumerics-amd-gpu-layout PUBLIC cxx_std_17)
target_include_directories(hostnumerics-amd-gpu-layout
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_sources(hostnumerics-amd-gpu-layout
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/amd_gpu_layout/mx.hpp"
)
set_target_properties(hostnumerics-amd-gpu-layout
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VERIFY_INTERFACE_HEADER_SETS ON
        VISIBILITY_INLINES_HIDDEN ON
)
if(HOSTNUMERICS_AMD_GPU_LAYOUT_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            hostnumerics-amd-gpu-layout
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        set_property(
            TARGET hostnumerics-amd-gpu-layout
            PROPERTY HOSTNUMERICS_USES_OPENMP ON
        )
    endif()
endif()

target_compile_features(hostnumerics PUBLIC cxx_std_20)
target_include_directories(hostnumerics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(hostnumerics
    PUBLIC
        roc::hostnumerics-core
)
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hostnumerics PRIVATE OpenMP::OpenMP_CXX)
    set_property(
        TARGET hostnumerics
        PROPERTY HOSTNUMERICS_USES_OPENMP ON
    )
endif()

target_sources(hostnumerics
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/epilogue.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/layer_norm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/mx.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/operation_types.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/reduction.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/softmax.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/structured_sparsity.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/tensor_operations.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/hostnumerics/validation.hpp"
)
set_target_properties(hostnumerics
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

# Enables the optional BLAS-backed GEMM implementation. This function is
# intentionally callable by an enclosing source build after the base target
# has already been added.
function(hostnumerics_enable_blas_backend)
    if(TARGET hostnumerics-blas)
        return()
    endif()

    set(_hostnumerics_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    list(PREPEND CMAKE_MODULE_PATH "${_hostnumerics_source_dir}/cmake")
    find_package(CBLAS REQUIRED MODULE)

    add_library(hostnumerics-blas STATIC
        "${_hostnumerics_source_dir}/src/backends/blas.cpp"
    )
    add_library(roc::hostnumerics-blas ALIAS hostnumerics-blas)
    target_compile_features(hostnumerics-blas PUBLIC cxx_std_20)
    target_include_directories(hostnumerics-blas
        PRIVATE
            "${_hostnumerics_source_dir}/src"
    )
    target_link_libraries(hostnumerics-blas
        PUBLIC
            roc::hostnumerics
        PRIVATE
            CBLAS::CBLAS
    )
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(hostnumerics-blas PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_sources(hostnumerics-blas
        PUBLIC
            FILE_SET HEADERS
            BASE_DIRS "${_hostnumerics_source_dir}/include"
            FILES
                "${_hostnumerics_source_dir}/include/roc/hostnumerics/backends/blas.hpp"
    )
    set_target_properties(hostnumerics-blas
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            POSITION_INDEPENDENT_CODE ON
            VISIBILITY_INLINES_HIDDEN ON
    )
    hostnumerics_install_component_target(
        hostnumerics-blas
        hostnumerics-blas-targets
        HostNumericsBLASTargets.cmake
    )
endfunction()

if(HOSTNUMERICS_BUILD_BLAS_BACKEND)
    hostnumerics_enable_blas_backend()
endif()

if(HOSTNUMERICS_BUILD_TESTING)
    enable_testing()
endif()

if(HOSTNUMERICS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(HOSTNUMERICS_BUILD_TESTING)
    add_subdirectory(tests)
    add_test(
        NAME hostnumerics-package-components
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
        hostnumerics-package-components
        PROPERTIES
            RUN_SERIAL TRUE
    )
endif()

hostnumerics_install_component_target(
    hostnumerics-core
    hostnumerics-core-targets
    HostNumericsCoreTargets.cmake
)
hostnumerics_install_component_target(
    hostnumerics
    hostnumerics-operations-targets
    HostNumericsOperationsTargets.cmake
)
hostnumerics_install_component_target(
    hostnumerics-amd-gpu-layout
    hostnumerics-amd-gpu-layout-targets
    HostNumericsAMDGPULayoutTargets.cmake
)

set_property(
    GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_property(
    GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}"
)
set_property(
    GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_INSTALL_CMAKEDIR
    "${HOSTNUMERICS_INSTALL_CMAKEDIR}"
)
set_property(
    GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_VERSION
    "${PROJECT_VERSION}"
)

# Verify that generic numerical targets do not depend on the AMD-specific leaf.
function(hostnumerics_assert_generic_target_boundary target_name)
    if(NOT TARGET "${target_name}")
        return()
    endif()
    foreach(property_name LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(link_libraries "${target_name}" "${property_name}")
        if(link_libraries
           AND link_libraries MATCHES "hostnumerics-amd-gpu-layout")
            message(FATAL_ERROR
                "Generic HostNumerics target ${target_name} depends on "
                "the AMD GPU layout module through ${property_name}."
            )
        endif()
    endforeach()
endfunction()

# Finalize package metadata after parent projects may enable optional backends.
function(hostnumerics_finalize_package)
    get_property(
        _hostnumerics_source_dir
        GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_SOURCE_DIR
    )
    get_property(
        _hostnumerics_binary_dir
        GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_BINARY_DIR
    )
    get_property(
        _hostnumerics_install_cmakedir
        GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_INSTALL_CMAKEDIR
    )
    get_property(
        _hostnumerics_version
        GLOBAL PROPERTY HOSTNUMERICS_PACKAGE_VERSION
    )

    set(HOSTNUMERICS_PACKAGE_HAS_BLAS OFF)
    if(TARGET hostnumerics-blas)
        set(HOSTNUMERICS_PACKAGE_HAS_BLAS ON)
    endif()
    set(HOSTNUMERICS_PACKAGE_OPERATIONS_HAS_OPENMP OFF)
    get_target_property(
        _hostnumerics_uses_openmp
        hostnumerics
        HOSTNUMERICS_USES_OPENMP
    )
    if(_hostnumerics_uses_openmp)
        set(HOSTNUMERICS_PACKAGE_OPERATIONS_HAS_OPENMP ON)
    endif()

    set(HOSTNUMERICS_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP OFF)
    get_target_property(
        _hostnumerics_amd_gpu_layout_uses_openmp
        hostnumerics-amd-gpu-layout
        HOSTNUMERICS_USES_OPENMP
    )
    if(_hostnumerics_amd_gpu_layout_uses_openmp)
        set(HOSTNUMERICS_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP ON)
    endif()

    foreach(generic_target
            hostnumerics-core
            hostnumerics
            hostnumerics-blas)
        hostnumerics_assert_generic_target_boundary("${generic_target}")
    endforeach()

    configure_package_config_file(
        "${_hostnumerics_source_dir}/cmake/HostNumericsConfig.cmake.in"
        "${_hostnumerics_binary_dir}/HostNumericsConfig.cmake"
        INSTALL_DESTINATION "${_hostnumerics_install_cmakedir}"
    )
    write_basic_package_version_file(
        "${_hostnumerics_binary_dir}/HostNumericsConfigVersion.cmake"
        VERSION "${_hostnumerics_version}"
        COMPATIBILITY SameMajorVersion
    )
endfunction()

cmake_language(
    DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
    CALL hostnumerics_finalize_package
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/HostNumericsConfigVersion.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCBLAS.cmake"
    DESTINATION "${HOSTNUMERICS_INSTALL_CMAKEDIR}"
    COMPONENT devel
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/HostNumerics"
    COMPONENT docs
)
