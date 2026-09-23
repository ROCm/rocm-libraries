# ########################################################################
# Copyright 2019-2026 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# --------------------------------------
# Update these variables at release time
#
# Set the library version
# NOTE(hipccl2): guarded so the hipccl2 root CMakeLists.txt can force
# VERSION_STRING via a CACHE variable before add_subdirectory(rocthrust). The
# fallback below is also hipccl-flavored (matching hipccl2's/hipccl3's root
# HIPCCL_VERSION placeholder, currently 0.1.0) rather than rocThrust's own
# 4.7.0, so a fully standalone build of this copy still reports a
# hipccl-consistent version. Duplicates the version number in multiple
# places - keep them in sync manually until real unified-versioning infra
# exists.
if(NOT DEFINED VERSION_STRING)
  set(VERSION_STRING "0.1.0")
endif()
# Set the minimum required rocPRIM version
set(MIN_ROCPRIM_PACKAGE_VERSION "4.0.0" CACHE STRING "Minimum version of rocPRIM to search for when ROCPRIM_FETCH_METHOD is set to PACKAGE.")
# Set the minimum required rocRAND version
set(MIN_ROCRAND_PACKAGE_VERSION "5.0.0" CACHE STRING "Minimum version of rocRAND to search for when ROCRAND_FETCH_METHOD is set to PACKAGE.")
# Set download branch for dependencies rocPRIM and rocRAND
set(ROCM_DEP_RELEASE_BRANCH "release/therock-10.0" CACHE STRING "Download branch for ROCm dependencies")
# --------------------------------------

# Install prefix
if(WIN32)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/package CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# NOTE(hipccl2): defaults the header install directory to the unified hipccl
# layout even for a fully standalone build of this copy. Set before
# project()/GNUInstallDirs claim the cache variable with the plain "include"
# default, so this wins first - matching the CMAKE_INSTALL_PREFIX pattern
# above. hipccl2's root CMakeLists.txt also forces this same value
# (CACHE ... FORCE) before add_subdirectory(rocthrust); through the
# superbuild, that FORCE already claimed it, so this line is a harmless no-op
# there.
set(CMAKE_INSTALL_INCLUDEDIR "include/hipccl" CACHE PATH "Install path for headers, relative to CMAKE_INSTALL_PREFIX")

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

# NOTE(hipccl2): two extra ".." vs upstream (rocm-libraries/projects/rocthrust)
# - this copy lives two directory levels deeper
# (rocm-libraries/projects/hipccl/hipccl2/rocthrust). Without this, the
# ${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake existence check in
# test/CMakeLists.txt silently resolves to a nonexistent path and skips the
# shared CTest category labeling with no error.
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
  set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")
endif()

# NOTE(hipccl router): unlike rocPRIM, rocThrust had no "am I nested"
# detection at all - it always called rocm_create_package() unconditionally.
# That is safe on its own, but fatal if rocThrust is add_subdirectory()'d
# alongside another project that also calls rocm_create_package() in the
# same configure (each triggers include(CPack), and a second include(CPack)
# fails trying to redefine the "package" target) - exactly what the
# projects/hipccl/CMakeLists.txt HIPCCL_BUILD_LEGACY route does by nesting
# rocprim + hipcub + rocthrust together. Guarded the same way as rocPRIM,
# including the NOT DEFINED wrapper so a parent can still opt back in.
# Standalone builds (nothing pre-set) are unaffected - always top-level,
# packaging always runs, exactly as before.
if(NOT DEFINED ROCTHRUST_PROJECT_IS_TOP_LEVEL)
  if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(ROCTHRUST_PROJECT_IS_TOP_LEVEL TRUE)
  else()
    set(ROCTHRUST_PROJECT_IS_TOP_LEVEL FALSE)
  endif()
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

include(CheckCXXCompilerFlag)

if(BUILD_OFFLOAD_COMPRESS)
  check_cxx_compiler_flag("--offload-compress -x hip" CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
  if(CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    # Some test will raise an unused-command-line-argument warning while building when they have no kernel code.
    # NOTE(hipccl2): switched from `set(CMAKE_CXX_FLAGS ...)` to
    # add_compile_options() - same effective scope (this directory and
    # everything add_subdirectory()'d after this point), but silences
    # ROCMChecks' "set on a target where possible" warning. Matches hipCUB's
    # own (pre-existing) pattern for the same flag.
    add_compile_options("--offload-compress" "-Wno-unused-command-line-argument")
  else()
    message(STATUS "Warning: BUILD_OFFLOAD_COMPRESS=ON but flag not supported by compiler. Ignoring option.")
  endif()
endif()

# Allow the user to optionally select offset type dispatch to fixed 32 or 64 bit types
set(THRUST_DISPATCH_TYPE "Dynamic" CACHE STRING "Select Thrust offset type dispatch." FORCE)
set_property(CACHE THRUST_DISPATCH_TYPE PROPERTY STRINGS "Dynamic" "Force32bit" "Force64bit")

#Adding CMAKE_PREFIX_PATH
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm ${ROCM_PATH} )

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip # FindHIP.cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CODE_COVERAGE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# rocm-cmake contains common cmake code for rocm projects to help
# setup and install
include( cmake/FindROCMCmake.cmake )
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMHeaderWrapper )
include( ROCMCheckTargetIds )
include( ROCMClients )

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

# Set up options for obtaining dependencies rocPRIM and rocRAND.
# PACKAGE: Search for an install package that contains the dependency.
# MONOREPO: Assume this is a monorepo checkout and search for the dependency in the directory at ../../projects/.
# DOWNLOAD: Download the dependency from the monorepo.
set(FETCH_METHOD_OPTIONS "PACKAGE" "MONOREPO" "DOWNLOAD")

set(ROCPRIM_FETCH_METHOD "PACKAGE" CACHE STRING "How to obtain the rocPRIM dependency")
set(ROCRAND_FETCH_METHOD "PACKAGE" CACHE STRING "How to obtain the rocRAND dependency")

# This function checks to see if the option variable it's passed is defined, and contains a valid value.
# If it does not contain a valid value, it issues a fatal failure with an error message.
function(check_option valid_options sel_option)
  if (DEFINED ${sel_option} AND NOT ${${sel_option}} IN_LIST ${valid_options})
    message(FATAL_ERROR "Unrecognized ${sel_option}: \"${${sel_option}}\". Valid options are: ${${valid_options}}.")
  endif()
endfunction()

check_option(FETCH_METHOD_OPTIONS ROCPRIM_FETCH_METHOD)
check_option(FETCH_METHOD_OPTIONS ROCRAND_FETCH_METHOD)

# Get dependencies
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
check_option(THRUST_HOST_SYSTEM_OPTIONS THRUST_HOST_SYSTEM)
add_compile_definitions(THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${THRUST_HOST_SYSTEM})

# If we're not grafting existing CUDA onto binaries, we must dissallow CUDA backend.
if (NOT GRAFT_THRUST_ONTO_BINARIES)
  set(THRUST_DEVICE_SYSTEM_OPTIONS OMP TBB CPP HIP)
  set(THRUST_DEVICE_SYSTEM HIP CACHE STRING "The host backend to target.")
  check_option(THRUST_DEVICE_SYSTEM_OPTIONS THRUST_DEVICE_SYSTEM)
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
  # NOTE(hipccl2): switched from `set(CMAKE_CXX_FLAGS ...)` to
  # add_compile_options() for the same reason as the offload-compress flag
  # above - silences ROCMChecks' "set on a target where possible" warning
  # without changing scope (still applies to this directory and everything
  # add_subdirectory()'d after this point, i.e. test/benchmark).
  add_compile_options("-O0" "-fprofile-instr-generate" "-fcoverage-mapping")
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
# NOTE(hipccl2): standalone packaging (rocm_create_package(), which triggers
# include(CPack), and the rocprim-dev/rocprim-devel cross-dependency that
# only makes sense alongside it) is intentionally disabled for this hipccl2
# copy of rocthrust, unconditionally - not just when nested under the
# hipccl2 superbuild. A standalone "rocthrust" package built from inside
# hipccl2/rocthrust would be a different, differently-named artifact than the
# unified "hipccl" package hipccl2's root CMakeLists.txt produces, installable
# side-by-side by a package manager that has no idea they overlap - risking
# duplicate/conflicting installs of the same headers. Build through hipccl2's
# root CMakeLists.txt (which calls rocm_create_package() once, unconditionally)
# to get a real distributable package; this copy only supports
# configure/build/`cmake --install`, matching hipccl3's own copy of
# rocThrust. The original (pre-guard) logic is preserved in git history if
# ever needed. rocm-libraries' own projects/rocthrust is untouched and still
# packages standalone exactly as before.

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()
