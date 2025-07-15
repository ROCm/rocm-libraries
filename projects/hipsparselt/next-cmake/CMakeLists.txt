# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.2)
project(
  hipsparselt
  VERSION 0.2.4
  LANGUAGES CXX)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # FetchContent timestamp handling
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)  # option() honors normal variables
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set cpack license file here to avoid rocm-cmake warning
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

include(FetchROCmCMake)
include(hipSPARSELtSupportedArchitectures)
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

rocm_setup_version(VERSION ${PROJECT_VERSION})
set(hipsparselt_SOVERSION ${PROJECT_VERSION_MAJOR})

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(WIN32)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path prefix." FORCE)
  else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix." FORCE)
  endif()
endif()

# Build Options and Validation
# ============================================================================
option(HIPSPARSELT_ENABLE_CLIENT "Build hipSPARSELt clients." ON)
option(HIPSPARSELT_ENABLE_CUDA "Build hipSPARSELt with CUDA backend." OFF)
option(HIPSPARSELT_ENABLE_HIP "Build hipSPARSELt with HIP backend." ON)

option(HIPSPARSELT_BUILD_SHARED_LIBS "Build the hipSPARSELt shared library." ON)
option(HIPSPARSELT_BUILD_TESTING "Build test client." ON)
option(HIPSPARSELT_BUILD_COVERAGE "Build tests with coverage enabled." OFF)

if(HIPSPARSELT_ENABLE_CLIENT)
  option(HIPSPARSELT_ENABLE_BENCHMARKS "Build benchmark client." ON)
  option(HIPSPARSELT_ENABLE_SAMPLES "Build client samples." ON)
  option(HIPSPARSELT_ENABLE_FORTRAN "Build Fortran clients." OFF)
  option(HIPSPARSELT_ENABLE_BLIS "Enable BLIS support for reference implementations." ON)
  # rocm-smi is not presently available on Windows
  cmake_dependent_option(HIPSPARSELT_ENABLE_ROCM_SMI "Require rocm_smi." ON "NOT WIN32" OFF)
endif()

option(HIPSPARSELT_ENABLE_MARKER "Enable rocTracer marker support." OFF)
option(HIPSPARSELT_ENABLE_ASAN "Build with address sanitizer enabled." OFF)
option(HIPSPARSELT_ENABLE_VERBOSE "Output additional build information." OFF)

set(HIPSPARSELT_HIPBLASLT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../hipblaslt/next-cmake" CACHE PATH "Path to hipblaslt directory.")
set(HIPSPARSELT_COVERAGE_GTEST_FILTER "*quick*:*pre_checkin*:*ILP64*-*known_bug*" 
    CACHE STRING "GTest filter for coverage tests.")

if(NOT HIPSPARSELT_ENABLE_CUDA)
    set(GPU_TARGETS "" CACHE STRING "AMD GFX targets to cross-compile")

    if(NOT GPU_TARGETS OR GPU_TARGETS STREQUAL "all")
        hipsparselt_get_base_architectures(GPU_TARGETS)
    else()
        hipsparselt_validate_gpu_targets("${GPU_TARGETS}")
    endif()
    message(STATUS "Building for GPU targets: ${GPU_TARGETS}")
endif()

if(HIPSPARSELT_ENABLE_FORTRAN AND HIPSPARSELT_ENABLE_ASAN)
    message(FATAL_ERROR "Cannot enable fortran clients and address sanitizers.")
endif()

if(HIPSPARSELT_BUILD_COVERAGE)
    if(HIPSPARSELT_ENABLE_FORTRAN)
        message(FATAL_ERROR "Cannot enable fortran clients and code coverage.")
    elseif(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(FATAL_ERROR "Code coverage is only supported for CMAKE_BUILD_TYPE=Debug.")
    endif()
endif()

if(HIPSPARSELT_ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# Dependency Setup
# ============================================================================
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(hip REQUIRED)

# if(HIPSPARSELT_ENABLE_HIP)
#     if(NOT ROCM_SUPERBUILD)
#         # TODO: This dependency shouldn't be required, why is it?
#         # find_package(hipsparse REQUIRED)
#     endif()
# endif()

if(HIPSPARSELT_ENABLE_ROCM_SMI)
    find_package(rocm_smi REQUIRED)
endif()

if(HIPSPARSELT_ENABLE_MARKER)
  find_library(ROCTX64_LIBRARY REQUIRED NAMES roctx64 libroctx64)
endif()

if(HIPSPARSELT_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_library(CUSPARSELT_LIBRARY 
        NAMES cusparseLt
        PATHS 
            ${CUDAToolkit_LIBRARY_ROOT}
        HINTS
            lib64
            lib
        REQUIRED
    )
    message(STATUS "Found cuSPARSELt: ${CUSPARSELT_LIBRARY}")
endif()

# Add Subdirectories
# ============================================================================
if(HIPSPARSELT_ENABLE_HIP)
    set(HIPBLASLT_PATH "${HIPSPARSELT_HIPBLASLT_PATH}")
    if(NOT EXISTS "${HIPBLASLT_PATH}")
        message(FATAL_ERROR "hipBLASLt not found at ${HIPBLASLT_PATH}. Please ensure hipblaslt is available.")
    endif()

    block(SCOPE_FOR VARIABLES)
        set(HIPBLASLT_ENABLE_HOST OFF CACHE BOOL "" FORCE)
        set(HIPBLASLT_ENABLE_CLIENT OFF CACHE BOOL "" FORCE)
        set(HIPBLASLT_ENABLE_DEVICE ON CACHE BOOL "" FORCE)
        set(TENSILELITE_ENABLE_HOST ON CACHE BOOL "" FORCE)
        set(HIPBLASLT_LIBLOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../library/src/hcc_detail/rocsparselt/src/spmm/Tensile/Logic/asm_full" CACHE STRING "" FORCE)

        if(WIN32)
            set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}hipsparselt/bin" CACHE PATH "path to tensile library" FORCE)
        else()
            set(HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipsparselt" CACHE PATH "path to tensile library" FORCE)
        endif()
        
        # Exposes targets for tensile-host and library
        # TODO: update the build output directory for hipblaslt/tensilelite
        add_subdirectory("${HIPBLASLT_PATH}" hipblaslt-BUILD)
    endblock()
endif()

add_subdirectory(library)

if(HIPSPARSELT_ENABLE_CLIENT)
    add_subdirectory(clients)
endif()

# Package Configuration
# ============================================================================
# # Set package name based on backend
# if(HIPSPARSELT_ENABLE_CUDA)
#     set(PACKAGE_NAME "hipsparselt-cuda")
# else()
#     set(PACKAGE_NAME "hipsparselt")
# endif()

# # Package dependencies
# if(NOT HIPSPARSELT_ENABLE_CUDA)
#     rocm_package_add_dependencies(DEPENDS "hipsparse >= 2.1.0")
# endif()

# # Configure package
# rocm_create_package(
#     NAME ${PACKAGE_NAME}
#     DESCRIPTION "ROCm Structured Sparsity Matrix Multiplication marshalling library"
#     MAINTAINER "hipSPARSELt Maintainer <hipsparselt-maintainer@amd.com>"
#     LDCONFIG
#     LDCONFIG_DIR ${CMAKE_INSTALL_FULL_LIBDIR}
# )

# Coverage Setup
# ============================================================================
if(HIPSPARSELT_BUILD_COVERAGE)
    target_compile_options(hipsparselt PRIVATE 
        -fprofile-arcs 
        -ftest-coverage
    )
    target_link_libraries(hipsparselt PRIVATE 
        --coverage 
        -lgcov
    )
endif()