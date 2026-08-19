# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(ROCHostValidation VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(HOST_VALIDATION_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostValidation"
)

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
option(HOST_VALIDATION_BUILD_BENCHMARKS
       "Build host-validation development benchmarks."
       OFF)
option(HOST_VALIDATION_ENABLE_FETCH
       "Allow host-validation tests to fetch missing dependencies."
       ${HOST_VALIDATION_STANDALONE})
option(HOST_VALIDATION_AMD_GPU_LAYOUT_ENABLE_OPENMP
       "Use OpenMP for large AMD GPU physical-layout transforms when available."
       ON)

# Install one public target and its component-specific export file.
function(
    roc_host_validation_install_component_target
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
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCHostValidation"
        NAMESPACE roc::
        COMPONENT devel
    )
endfunction()

add_library(host-validation-core INTERFACE)
add_library(roc::host-validation-core ALIAS host-validation-core)

add_library(host-validation-amd-gpu-layout STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/amd_gpu_layout/mx.cpp"
)
add_library(
    roc::host-validation-amd-gpu-layout
    ALIAS host-validation-amd-gpu-layout
)

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

add_library(host-validation-blocked STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/blocked.cpp"
)
add_library(roc::host-validation-blocked ALIAS host-validation-blocked)

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
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/detail/scalar_codec.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/index_order.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/scalar.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/tensor.hpp"
)

target_compile_features(host-validation-amd-gpu-layout PUBLIC cxx_std_17)
target_include_directories(host-validation-amd-gpu-layout
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(host-validation-amd-gpu-layout
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/amd_gpu_layout/mx.hpp"
)
set_target_properties(host-validation-amd-gpu-layout
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VERIFY_INTERFACE_HEADER_SETS ON
        VISIBILITY_INLINES_HIDDEN ON
)
if(HOST_VALIDATION_AMD_GPU_LAYOUT_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            host-validation-amd-gpu-layout
            PRIVATE
                OpenMP::OpenMP_CXX
        )
        set_property(
            TARGET host-validation-amd-gpu-layout
            PROPERTY ROC_HOST_VALIDATION_USES_OPENMP ON
        )
    endif()
endif()

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
find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(host-validation PRIVATE OpenMP::OpenMP_CXX)
    set_property(
        TARGET host-validation
        PROPERTY ROC_HOST_VALIDATION_USES_OPENMP ON
    )
endif()

target_sources(host-validation
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/axpby.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/comparison.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/epilogue.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/generation.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/generation_compatibility.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/generation_primitives.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/gemm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/layer_norm.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/operation_types.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/reduction.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/softmax.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/structured_sparsity.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/validation.hpp"
)
set_target_properties(host-validation
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        POSITION_INDEPENDENT_CODE ON
        VISIBILITY_INLINES_HIDDEN ON
)

target_compile_features(host-validation-blocked PUBLIC cxx_std_20)
target_link_libraries(host-validation-blocked PUBLIC roc::host-validation)
target_include_directories(host-validation-blocked
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_sources(host-validation-blocked
    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/include/roc/host_validation/backends/blocked.hpp"
)
set_target_properties(host-validation-blocked
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
    list(PREPEND CMAKE_MODULE_PATH "${_host_validation_source_dir}/cmake")
    find_package(CBLAS REQUIRED MODULE)

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
        PRIVATE
            CBLAS::CBLAS
    )
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
            ROC_HOST_VALIDATION_CBLAS_BUILD_PROVIDER "${CBLAS_BUILD_PROVIDER}"
            ROC_HOST_VALIDATION_CBLAS_INTEGER_SIZE "${CBLAS_INTEGER_SIZE}"
            VISIBILITY_INLINES_HIDDEN ON
    )
    roc_host_validation_install_component_target(
        host-validation-blas
        host-validation-blas-targets
        ROCHostValidationBLASTargets.cmake
    )
endfunction()

if(HOST_VALIDATION_BUILD_BLAS_BACKEND)
    roc_host_validation_enable_blas_backend()
endif()

# Add the MX generation implementation to the current build.
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
    roc_host_validation_install_component_target(
        host-validation-mx
        host-validation-mx-targets
        ROCHostValidationMXTargets.cmake
    )
endfunction()

# MX scalar/layout types are part of Core, and the compiled MX generator has no
# mandatory external dependency. Define it consistently in every source build
# so MX support cannot become an untested configuration variant.
roc_host_validation_enable_mx_backend()

if(HOST_VALIDATION_BUILD_TESTING)
    enable_testing()
endif()

if(HOST_VALIDATION_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(HOST_VALIDATION_BUILD_TESTING)
    add_subdirectory(tests)
    add_test(
        NAME host-validation-package-components
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
        host-validation-package-components
        PROPERTIES
            RUN_SERIAL TRUE
    )
endif()

roc_host_validation_install_component_target(
    host-validation-core
    host-validation-core-targets
    ROCHostValidationCoreTargets.cmake
)
roc_host_validation_install_component_target(
    host-validation
    host-validation-operations-targets
    ROCHostValidationOperationsTargets.cmake
)
roc_host_validation_install_component_target(
    host-validation-blocked
    host-validation-blocked-targets
    ROCHostValidationBlockedTargets.cmake
)
roc_host_validation_install_component_target(
    host-validation-amd-gpu-layout
    host-validation-amd-gpu-layout-targets
    ROCHostValidationAMDGPULayoutTargets.cmake
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

# Verify that generic numerical targets do not depend on the AMD-specific leaf.
function(roc_host_validation_assert_generic_target_boundary target_name)
    if(NOT TARGET "${target_name}")
        return()
    endif()
    foreach(property_name LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(link_libraries "${target_name}" "${property_name}")
        if(link_libraries
           AND link_libraries MATCHES "host-validation-amd-gpu-layout")
            message(FATAL_ERROR
                "Generic host-validation target ${target_name} depends on "
                "the AMD GPU layout module through ${property_name}."
            )
        endif()
    endforeach()
endfunction()

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
    set(HOST_VALIDATION_PACKAGE_BLAS_BUILD_PROVIDER)
    set(HOST_VALIDATION_PACKAGE_BLAS_INTEGER_SIZE)
    if(TARGET host-validation-blas)
        set(HOST_VALIDATION_PACKAGE_HAS_BLAS ON)
        get_target_property(
            HOST_VALIDATION_PACKAGE_BLAS_BUILD_PROVIDER
            host-validation-blas
            ROC_HOST_VALIDATION_CBLAS_BUILD_PROVIDER
        )
        get_target_property(
            HOST_VALIDATION_PACKAGE_BLAS_INTEGER_SIZE
            host-validation-blas
            ROC_HOST_VALIDATION_CBLAS_INTEGER_SIZE
        )
    endif()
    if(NOT TARGET host-validation-mx)
        message(FATAL_ERROR
            "The required host-validation MX generation target is missing."
        )
    endif()
    set(HOST_VALIDATION_PACKAGE_HAS_MX ON)

    set(HOST_VALIDATION_PACKAGE_OPERATIONS_HAS_OPENMP OFF)
    get_target_property(
        _host_validation_uses_openmp
        host-validation
        ROC_HOST_VALIDATION_USES_OPENMP
    )
    if(_host_validation_uses_openmp)
        set(HOST_VALIDATION_PACKAGE_OPERATIONS_HAS_OPENMP ON)
    endif()

    set(HOST_VALIDATION_PACKAGE_MX_HAS_OPENMP OFF)
    if(TARGET host-validation-mx)
        get_target_property(
            _host_validation_mx_uses_openmp
            host-validation-mx
            ROC_HOST_VALIDATION_USES_OPENMP
        )
        if(_host_validation_mx_uses_openmp)
            set(HOST_VALIDATION_PACKAGE_MX_HAS_OPENMP ON)
        endif()
    endif()

    set(HOST_VALIDATION_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP OFF)
    get_target_property(
        _host_validation_amd_gpu_layout_uses_openmp
        host-validation-amd-gpu-layout
        ROC_HOST_VALIDATION_USES_OPENMP
    )
    if(_host_validation_amd_gpu_layout_uses_openmp)
        set(HOST_VALIDATION_PACKAGE_AMD_GPU_LAYOUT_HAS_OPENMP ON)
    endif()

    foreach(generic_target
            host-validation-core
            host-validation
            host-validation-blocked
            host-validation-blas
            host-validation-mx)
        roc_host_validation_assert_generic_target_boundary("${generic_target}")
    endforeach()

    configure_package_config_file(
        "${_host_validation_source_dir}/cmake/ROCHostValidationConfig.cmake.in"
        "${_host_validation_binary_dir}/ROCHostValidationConfig.cmake"
        INSTALL_DESTINATION "${_host_validation_install_cmakedir}"
    )
    configure_file(
        "${_host_validation_source_dir}/cmake/ROCHostValidationTargets.cmake.in"
        "${_host_validation_binary_dir}/ROCHostValidationTargets.cmake"
        @ONLY
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

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ROCHostValidationTargets.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCBLAS.cmake"
    DESTINATION "${HOST_VALIDATION_INSTALL_CMAKEDIR}"
    COMPONENT devel
)
install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/doc/ROCHostValidation"
    COMPONENT docs
)
