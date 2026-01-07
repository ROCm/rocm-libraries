# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json for clang tooling")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Load backward compatibility shim BEFORE project() to retain behaviour in included files.
include(rocsparse_shim)

# Do not set VERSION here, it is managed by rocm_setup_version
project(rocsparse LANGUAGES CXX C)

if(POLICY CMP0066)
    cmake_policy(SET CMP0066 NEW) # Honor per-config flags in try_compile()
endif()

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW) # option() honors normal variables
endif()

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW) # FetchContent timestamp handling
endif()

# Set cpack license file to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

include(fetch_rocm_cmake)
include(rocsparse_supported_architectures.cmake)
include(rocsparse_target_configure_sanitizers)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# Version setup
set(VERSION_STRING "4.3.0")
set(SOVERSION_STRING "1.0")
rocm_setup_version(VERSION ${VERSION_STRING})
set(rocsparse_SOVERSION ${SOVERSION_STRING})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix" FORCE)
    endif()
endif()

# --- Build Options ---

option(ROCSPARSE_BUILD_SHARED_LIBS "Build rocSPARSE as a shared library" ON)
option(ROCSPARSE_BUILD_TESTING "Build tests (requires googletest)" OFF)
option(ROCSPARSE_ENABLE_BENCHMARKS "Build benchmarks" OFF)
option(ROCSPARSE_ENABLE_SAMPLES "Build examples" OFF)
option(ROCSPARSE_BUILD_DOCS "Build documentation" OFF)

# Optional features
option(ROCSPARSE_BUILD_COVERAGE "Build with code coverage enabled" OFF)
option(ROCSPARSE_ENABLE_ASAN "Build with address sanitizer enabled" OFF)
option(ROCSPARSE_ENABLE_MEMSTAT "Build with memory statistics enabled" OFF)
option(ROCSPARSE_ENABLE_ILP64 "Build with rocsparse_int equal to int64_t" OFF)
option(ROCSPARSE_ENABLE_COMPRESSED_DBG "Enable compressed debug symbols" ON)
option(ROCSPARSE_ENABLE_ROCBLAS "Enable building rocSPARSE with rocBLAS" ON)
option(ROCSPARSE_ENABLE_MARKER "Enable rocTracer marker support" ON)
option(ROCSPARSE_ENABLE_OFFLOAD_COMPRESS "Enable offload compression during compilation" ON)

if(NOT ROCSPARSE_ENABLE_FORTRAN)
    option(ROCSPARSE_ENABLE_FORTRAN "Build rocSPARSE clients requiring Fortran capabilities" ON)
endif()

if(ROCSPARSE_ENABLE_SAMPLES OR ROCSPARSE_BUILD_TESTING OR ROCSPARSE_ENABLE_BENCHMARKS)
    set(ROCSPARSE_ENABLE_CLIENT ON)
else()
    set(ROCSPARSE_ENABLE_CLIENT OFF)
endif()

if(ROCSPARSE_ENABLE_ASAN AND ROCSPARSE_ENABLE_FORTRAN)
    message(FATAL_ERROR "Cannot enable both address sanitizer and Fortran clients")
endif()

if(ROCSPARSE_BUILD_COVERAGE AND ROCSPARSE_ENABLE_FORTRAN)
    message(FATAL_ERROR "Cannot enable both code coverage and Fortran clients")
endif()

if(ROCSPARSE_BUILD_COVERAGE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "'hipcc' or 'amdclang' compiler required to compile for ROCm platform")
endif()

if(ROCSPARSE_ENABLE_OFFLOAD_COMPRESS)
    check_cxx_compiler_flag("--offload-compress" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    if(NOT CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
        message(FATAL_ERROR
            "The option '--offload-compress' is not supported by the CXX compiler. "
            "Turn off ROCSPARSE_ENABLE_OFFLOAD_COMPRESS."
        )
    endif()
endif()

if(ROCSPARSE_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

set(GPU_TARGETS "all" CACHE STRING "AMD GFX targets to cross-compile")

if(NOT GPU_TARGETS)
    message(FATAL_ERROR "No GPU targets specified.")
elseif(GPU_TARGETS STREQUAL "all")
    rocsparse_get_base_architectures(GPU_TARGETS)
else()
    rocsparse_validate_gpu_targets("${GPU_TARGETS}")
endif()

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

# --- Dependency Setup ---

find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)

set(ROCSPARSE_MIN_ROCPRIM_VERSION "4.0.0" CACHE STRING "Minimum version of rocPRIM")
find_package(rocprim REQUIRED)
if(rocprim_VERSION VERSION_LESS ROCSPARSE_MIN_ROCPRIM_VERSION)
    message(FATAL_ERROR
        "Incompatible rocPRIM version. Found: ${rocprim_VERSION}, Required: ${ROCSPARSE_MIN_ROCPRIM_VERSION}"
    )
endif()

set(ROCSPARSE_MIN_ROCBLAS_VERSION "4.1.0" CACHE STRING "Minimum version of rocBLAS")
if(ROCSPARSE_ENABLE_ROCBLAS)
    find_package(rocblas REQUIRED)
    if(rocblas_VERSION VERSION_LESS ROCSPARSE_MIN_ROCBLAS_VERSION)
        message(FATAL_ERROR
            "Incompatible rocBLAS version. Found: ${rocblas_VERSION}, Required: ${ROCSPARSE_MIN_ROCBLAS_VERSION}"
        )
    endif()
endif()

if(ROCSPARSE_ENABLE_CLIENT)
    find_package(OpenMP QUIET)
    if(NOT OPENMP_FOUND)
        message(WARNING "OpenMP not found. Clients may have reduced performance.")
    endif()
endif()

if(ROCSPARSE_BUILD_SHARED_LIBS)
    add_library(rocsparse SHARED "")
else()
    add_library(rocsparse STATIC "")
endif()

add_library(roc::rocsparse ALIAS rocsparse)

add_subdirectory(library)

target_compile_features(rocsparse PRIVATE cxx_std_17)

if(ROCSPARSE_ENABLE_COMPRESSED_DBG)
    target_compile_options(rocsparse PRIVATE
        $<$<CONFIG:Debug>:-gz>
        $<$<CONFIG:RelWithDebInfo>:-gz>
    )
endif()

if(ROCSPARSE_ENABLE_OFFLOAD_COMPRESS)
    target_compile_options(rocsparse PRIVATE --offload-compress)
endif()

if(ROCSPARSE_BUILD_COVERAGE)
    target_compile_options(rocsparse PRIVATE -fprofile-arcs -ftest-coverage)
    target_link_options(rocsparse PRIVATE --coverage -lgcov)
endif()

if(ROCSPARSE_ENABLE_ASAN)
    target_compile_options(rocsparse PRIVATE -fsanitize=address -shared-libasan)
    target_link_options(rocsparse PRIVATE -fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

rocm_set_soversion(rocsparse ${rocsparse_SOVERSION})

if(ROCSPARSE_ENABLE_CLIENT)
    add_subdirectory(clients)
endif()

if(ROCSPARSE_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# --- Packaging and installation ---

if(ROCSPARSE_ENABLE_CLIENT)
    include(rocm_client_prerequisites)
    rocm_package_setup_component(clients)
    rocm_package_setup_client_component(clients-common)
endif()

if(ROCSPARSE_BUILD_TESTING OR BUILD_TESTING)
    rocm_package_setup_client_component(tests
        DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
    )
endif()

if(ROCSPARSE_ENABLE_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks
        DEPENDS
            COMPONENT clients-common
            DEB "${OPENMP_DEB}"
            RPM "${OPENMP_RPM}"
    )
endif()

if(ROCSPARSE_ENABLE_SAMPLES)
    rocm_package_setup_client_component(samples)
endif()

if(ROCSPARSE_ENABLE_FORTRAN)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
endif()

# Work around code object stripping failure if using /usr/bin/strip
find_program(LLVM_STRIP NAMES llvm-strip PATH_SUFFIXES lib/llvm/bin llvm/bin bin)
if(LLVM_STRIP)
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${LLVM_STRIP}")
else()
    message(
        WARNING
            "llvm-strip not found, using default strip tool. This may cause stripping to fail during packaging."
    )
endif()

rocm_install(
    TARGETS rocsparse
    EXPORT rocsparse-targets
    ARCHIVE COMPONENT "devel"
    LIBRARY COMPONENT "runtime"
    RUNTIME COMPONENT "runtime"
)

rocm_install(
    DIRECTORY "${PROJECT_BINARY_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT "devel"
)

rocm_export_targets(
    NAMESPACE roc::
    TARGETS rocsparse
    EXPORT rocsparse-targets
    DEPENDS
    PACKAGE hip
    PACKAGE rocprim
    COMPATIBILITY SameMajorVersion
    CONFIG_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocsparse-config.cmake.in"
)


# Set up runtime component
rocm_package_setup_component(runtime)

# HIP runtime dependencies
set(HIP_RUNTIME_MINIMUM "4.5.0")
if(ROCSPARSE_ENABLE_ASAN)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan")
else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd")
endif()

# Add runtime dependencies
rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}" COMPONENT runtime)

# rocBLAS dependency
if(rocblas_FOUND)
    rocm_package_add_dependencies(DEPENDS "rocblas >= ${ROCSPARSE_MIN_ROCBLAS_VERSION}" COMPONENT runtime)
    # Static library dependencies for rocBLAS
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${ROCSPARSE_MIN_ROCBLAS_VERSION}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${ROCSPARSE_MIN_ROCBLAS_VERSION}")
endif()

# Add rocm-core dependency
rocm_package_add_dependencies(DEPENDS "rocm-core" COMPONENT runtime)

# Static library dependencies for HIP
rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")

# Development package depends on runtime package
rocm_package_add_deb_dependencies(DEPENDS "rocsparse (>= ${rocsparse_VERSION})" COMPONENT devel)
rocm_package_add_rpm_dependencies(DEPENDS "rocsparse >= ${rocsparse_VERSION}" COMPONENT devel)

# CPack configuration
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")
set(ROCSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
    "Path placed into ldconfig file"
)

# Create package
rocm_create_package(
    NAME rocsparse
    DESCRIPTION "ROCm SPARSE library"
    MAINTAINER "rocSPARSE Maintainer <rocsparse-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCSPARSE_CONFIG_DIR}
)

# --- Coverage setup ---

if(ROCSPARSE_BUILD_COVERAGE)
    # Coverage analysis target
    add_custom_target(
        coverage_analysis
        COMMAND echo "Running coverage with GTEST_FILTER=\${GTEST_FILTER}"
        COMMAND ./clients/staging/rocsparse-test --gtest_filter="\${GTEST_FILTER}"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    add_dependencies(coverage_analysis rocsparse)

    # Coverage output preparation
    add_custom_target(
        coverage_output
        DEPENDS coverage_analysis
        COMMAND ${CMAKE_COMMAND} -E make_directory lcoverage
        COMMAND ${CMAKE_COMMAND} -E echo "#!/bin/bash" > llvm-gcov.sh
        COMMAND ${CMAKE_COMMAND} -E echo "# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
        COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
        COMMAND chmod +x llvm-gcov.sh
    )

    # Generate coverage report
    add_custom_command(
        TARGET coverage_output
        COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh
                --capture -o lcoverage/raw_main_coverage.info
        COMMAND lcov --remove lcoverage/raw_main_coverage.info
                "'${PROJECT_SOURCE_DIR}/clients/*'"
                "'${PROJECT_SOURCE_DIR}/build/*'"
                "'/opt/*'" "'/usr/*'"
                -o lcoverage/main_coverage.info
        COMMAND genhtml lcoverage/main_coverage.info --output-directory lcoverage
    )

    add_custom_target(coverage DEPENDS coverage_output)

    # Coverage cleanup
    add_custom_target(
        coverage_cleanup
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()
