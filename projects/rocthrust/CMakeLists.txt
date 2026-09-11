# ########################################################################
# Copyright 2019-2026 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# --------------------------------------
# Update these variables at release time
#
# Set the library version
set(VERSION_STRING "4.7.0")
# Set the minimum required rocPRIM version
set(MIN_ROCPRIM_PACKAGE_VERSION "4.0.0" CACHE STRING "Minimum version of rocPRIM to search for when ROCPRIM_FETCH_METHOD is set to PACKAGE.")
# Set download branch for dependencies rocPRIM and rocRAND
set(ROCM_DEP_RELEASE_BRANCH "release/therock-10.0" CACHE STRING "Download branch for ROCm dependencies")
# --------------------------------------

# Install prefix
if(WIN32)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/package CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Thrust project
# Note: C is required here for dependencies
project(rocthrust LANGUAGES CXX C)

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (CMAKE_CXX_STANDARD EQUAL 14)
  message(WARNING "C++14 will be deprecated in the next major release")
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# Set HIP flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

if(NOT DEFINED ROCM_LIBRARIES_ROOT)
  set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

include(CheckLanguage)
include(CMakeDependentOption)

# Build options
# Disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)
option(BUILD_TEST "Build tests" OFF)
option(CODE_COVERAGE "Enable code coverage" OFF)
option(BUILD_HIPSTDPAR_TEST "Build hipstdpar tests" OFF)
option(BUILD_HIPSTDPAR_TEST_WITH_TBB "Build hipstdpar tests with TBB" OFF)
option(BUILD_EXAMPLE "Build example" OFF)
option(BUILD_BENCHMARK "Build benchmark" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_OFFLOAD_COMPRESS "Build rocThrust with offload compression" ON)
option(ROCTHRUST_USE_LIBHIPCXX "Build rocThrust using libhipcxx backend, if available" ON)
option(ROCTHRUST_USE_LIBCUDACXX "Build rocThrust using libcudacxx backend, if available" ON)
cmake_dependent_option(ENABLE_UPSTREAM_TESTS "Enable upstream (thrust) tests" ON BUILD_TEST OFF)
cmake_dependent_option(USE_SYSTEM_LIB "Use existing system ROCm library installation when building tests" OFF BUILD_TEST OFF)
option(EXTERNAL_DEPS_FORCE_DOWNLOAD "Force download of non-ROCm dependencies (e.g., Google Test)" OFF)
option(LINK_HIP_DEVICE_LIBS "Links against HIP device libraries. Required for device-side acceleration." ON)

option(GRAFT_THRUST_ONTO_BINARIES "!!! TODO !!!" OFF)

if((BUILD_TEST OR BUILD_BENCHMARK OR BUILD_EXAMPLE) AND NOT (${LINK_HIP_DEVICE_LIBS}))
  message(FATAL_ERROR "rocThrust tests, benchmarks, and examples must be built with LINK_HIP_DEVICE_LIBS=ON since they require device acceleration.")
endif()

if(${LINK_HIP_DEVICE_LIBS})
  check_language(HIP)
endif()

cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)

# Allow the user to optionally select offset type dispatch to fixed 32 or 64 bit types
set(THRUST_DISPATCH_TYPE "Dynamic" CACHE STRING "Select Thrust offset type dispatch." FORCE)
set_property(CACHE THRUST_DISPATCH_TYPE PROPERTY STRINGS "Dynamic" "Force32bit" "Force64bit")

# Add project and ROCm-related CMake module paths.
list(APPEND CMAKE_MODULE_PATH
  # Shared within monorepo:
  ${CMAKE_CURRENT_SOURCE_DIR}/../../shared/cmake
  # ROCm related:
  $ENV{ROCM_PATH}
  $ENV{ROCM_PATH}/share
  $ENV{ROCM_PATH}/lib/cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CODE_COVERAGE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# rocm-cmake must be included before processing 'GPU_TARGETS'.
include(dependencies/rocm-cmake)

if(USE_HIPCXX)
  enable_language(HIP)
else()
  # Use target ID syntax if supported for GPU_TARGETS
  if (NOT DEFINED AMDGPU_TARGETS)
    set(GPU_TARGETS "all" CACHE STRING "GPU architectures to compile for")
  else()
    set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for")
  endif()
  set_property(CACHE GPU_TARGETS PROPERTY STRINGS "all")

  if(GPU_TARGETS STREQUAL "all")
    if(BUILD_ADDRESS_SANITIZER)
      # ASAN builds require xnack
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
        TARGETS "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+"
      )
    else()
      rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
        TARGETS "gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250"
      )
    endif()
    set(GPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "GPU architectures to compile for" FORCE)
  endif()
endif()

# HIP must be included before checking for 'OFFLOAD_COMPRESS'. This is because
# adding '--offload-compress' to a CXX target (i.e. non -xhip), causes compiler
# warnings that may be treated as error, and thus failing detection of certain
# depedencies in subsequent 'find_package'-calls.
if (NOT GRAFT_THRUST_ONTO_BINARIES)
  include(dependencies/hip)
endif()

# TODO: this should be moved to toolchain-*.cmake, but this requires a hard
# depedency of CMake HIP language support due to '--offload-compress' only
# working with '-xhip'.
include(flags/offload_compress)

# Other depedencies must be included after adding in offload compress to
# reduce the size of kernels contained in depedencies.
include(cmake/Dependencies.cmake)

# Verify that supported compilers are used
if (NOT WIN32 OR NOT GRAFT_THRUST_ONTO_BINARIES)
  include(cmake/VerifyCompiler.cmake)
endif()

set(RNG_SEED_COUNT 0 CACHE STRING "Number of true random sequences to test each input size for")
set(PRNG_SEEDS 1 CACHE STRING "Seeds of pseudo random sequences to test each input size for")

# Set host and device backend system options
set(THRUST_HOST_SYSTEM_OPTIONS CPP OMP TBB)
set(THRUST_HOST_SYSTEM CPP CACHE STRING "The host backend to target.")
if(NOT THRUST_HOST_SYSTEM IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "Unrecognized THRUST_HOST_SYSTEM: ${THRUST_DEVICE_SYSTEM}"
  )
endif()
add_compile_definitions(THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${THRUST_HOST_SYSTEM})

# If we're not grafting existing CUDA onto binaries, we must dissallow CUDA backend.
if (NOT GRAFT_THRUST_ONTO_BINARIES)
  set(THRUST_DEVICE_SYSTEM_OPTIONS OMP TBB CPP HIP)
  set(THRUST_DEVICE_SYSTEM HIP CACHE STRING "The host backend to target.")
  if(NOT THRUST_DEVICE_SYSTEM IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
    message(
      FATAL_ERROR
      "Unrecognized THRUST_DEVICE_SYSTEM: ${THRUST_DEVICE_SYSTEM}"
    )
  endif()
  add_compile_definitions(THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${THRUST_DEVICE_SYSTEM})
endif()

set(COMPILE_OPTIONS -Wall -Wextra)
if(NOT DISABLE_WERROR)
  list(APPEND COMPILE_OPTIONS -Werror)
endif()

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++17 is supported")
endif()

if (WIN32)
  add_compile_options(-xhip)
  add_compile_definitions(THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
endif()

if (ROCTHRUST_USE_LIBHIPCXX)
  add_compile_definitions(USE_LIBHIPCXX)
endif()

if (ROCTHRUST_USE_LIBCUDACXX)
  add_compile_definitions(USE_LIBCUDACXX)
endif()

# Address Sanitizer
if(BUILD_ADDRESS_SANITIZER)
  add_compile_definitions(ADDRESS_SANITIZER_BUILD)
  list(APPEND COMPILE_OPTIONS -fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address -shared-libasan -fuse-ld=lld)
endif()

# Setup VERSION
rocm_setup_version(VERSION ${VERSION_STRING})
math(EXPR rocthrust_VERSION_NUMBER "${rocthrust_VERSION_MAJOR} * 100000 + ${rocthrust_VERSION_MINOR} * 100 + ${rocthrust_VERSION_PATCH}")

# Thrust (header-only) library
if(GRAFT_THRUST_ONTO_BINARIES)
  # If we're grafting CCCL we don't need the actual rocThrust library!
  find_package(CCCL CONFIG REQUIRED)
else()
  add_subdirectory(thrust)
endif()

if(BUILD_TEST OR BUILD_BENCHMARK OR BUILD_HIPSTDPAR_TEST)
  rocm_package_setup_component(clients)
endif()

if(CODE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fprofile-instr-generate -fcoverage-mapping")
endif()

# Tests
if(BUILD_TEST AND USE_SYSTEM_LIB AND NOT GRAFT_THRUST_ONTO_BINARIES)
  find_package(rocprim REQUIRED CONFIG PATHS "/opt/rocm/rocprim")
  if (${rocprim_VERSION} VERSION_LESS ${MIN_ROCPRIM_PACKAGE_VERSION})
    message(WARNING "The installed rocprim version, ${rocprim_VERSION}, is less than the minimum required version ${MIN_ROCPRIM_PACKAGE_VERSION}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
  endif()
  find_package(rocthrust REQUIRED CONFIG PATHS "/opt/rocm/rocthrust")
  if (NOT ${rocthrust_VERSION} VERSION_EQUAL ${VERSION_STRING})
    message(WARNING "The installed rocthrust version, ${rocthrust_VERSION}, does not match project version ${VERSION_STRING}. Building tests with USE_SYSTEM_LIB=ON may not work properly.")
  endif()
endif()

if(BUILD_TEST OR BUILD_HIPSTDPAR_TEST)
  rocm_package_setup_client_component(tests)
  if (ENABLE_UPSTREAM_TESTS)
    enable_testing()
  endif()

  # Used by tests to define their RESOURCE_GROUPS
  set(DEFAULT_GPU_RESOURCE "gpus")

  # We still want the testing to be compiled to catch some errors
  #TODO: Get testing folder working with HIP on Windows
  if (NOT WIN32 AND BUILD_TEST)
    add_subdirectory(testing)
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

# Examples
if(BUILD_EXAMPLE)
  add_subdirectory(examples)
endif()

# Benchmark
if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

# Package
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

rocm_package_add_deb_dependencies(DEPENDS "rocprim-dev >= 2.10.1")
rocm_package_add_rpm_dependencies(DEPENDS "rocprim-devel >= 2.10.1")
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_DEBIAN_DEVEL_PACKAGE_PROVIDES "hipstdpar")
set(CPACK_RPM_DEVEL_PACKAGE_PROVIDES "hipstdpar")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RPM_PACKAGE_LICENSE "ASL 2.0")

# if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
#   set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
# endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

rocm_create_package(
  NAME rocthrust
  DESCRIPTION "rocThrust is a ROCm port of the Thrust library, written in HIP"
  MAINTAINER "rocthrust-maintainer@amd.com"
  HEADER_ONLY
)

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()
