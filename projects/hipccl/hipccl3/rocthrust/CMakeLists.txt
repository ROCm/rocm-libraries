# ########################################################################
# Copyright 2019-2020 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# Install prefix
if(WIN32)
  set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/package CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Thrust project
project(rocthrust LANGUAGES CXX)

#Adding CMAKE_PREFIX_PATH
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# AMD targets
set(AMDGPU_TARGETS gfx900:xnack-;gfx906:xnack-;gfx908:xnack- CACHE STRING "List of specific machine types for library to target")

# Verify that hcc or hipcc compiler is used on ROCM platform
include(cmake/VerifyCompiler.cmake)

# Build options
# Disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(DOWNLOAD_ROCPRIM "Download rocPRIM and do not search for rocPRIM package" OFF)
set(RNG_SEED_COUNT 0 CACHE STRING "Number of true random sequences to test each input size for")
set(PRNG_SEEDS 1 CACHE STRING "Seeds of pseudo random sequences to test each input size for")

set(THRUST_HOST_SYSTEM_OPTIONS CPP OMP TBB)
set(THRUST_HOST_SYSTEM CPP CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_HOST_SYSTEM
  PROPERTY STRINGS ${THRUST_HOST_SYSTEM_OPTIONS}
)

if (NOT THRUST_HOST_SYSTEM IN_LIST THRUST_HOST_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_HOST_SYSTEM must be one of ${THRUST_HOST_SYSTEM_OPTIONS}"
  )
endif ()

# Set CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Get dependencies
include(cmake/Dependencies.cmake)

# Setup VERSION
rocm_setup_version(VERSION "2.10.7")

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

# Thrust (with HIP backend)
add_subdirectory(thrust)

# Tests
if(BUILD_TEST)
  # We still want the testing to be compiled to catch some errors
  add_subdirectory(testing)
  enable_testing()
  add_subdirectory(test)
endif()

# Examples
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Benchmarks
if(BUILD_BENCHMARKS)
  add_subdirectory(internal/benchmark)
endif()

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

# Package
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocprim (>= 2.10.1)") # 1.5.18263 is HIP version in ROCm 1.8.2
set(CPACK_DEBIAN_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_CONFLICTS "hip-thrust, thrust")
set(CPACK_RPM_PACKAGE_REQUIRES "rocprim >= 2.10.1")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

rocm_create_package(
  NAME rocthrust
  DESCRIPTION "Radeon Open Compute Thrust library"
  MAINTAINER "rocthrust-maintainer@amd.com"
)

# List of tests that aren't implemented for all backends, but are implemented for CUDA.
set(THRUST_PARTIALLY_IMPLEMENTED_CUDA
    async_copy
    async_for_each
    async_reduce
    async_reduce_into
    async_sort
    async_transform
    event
    future
)

# List of tests that aren't implemented for all backends, but are implemented for CPP.
set(THRUST_PARTIALLY_IMPLEMENTED_CPP
)

# List of tests that aren't implemented for all backends, but are implemented for TBB.
set(THRUST_PARTIALLY_IMPLEMENTED_TBB
)

# List of tests that aren't implemented for all backends, but are implemented for OMP.
set(THRUST_PARTIALLY_IMPLEMENTED_OMP
)

# List of all partially implemented tests.
set(THRUST_PARTIALLY_IMPLEMENTED
  ${THRUST_PARTIALLY_IMPLEMENTED_CUDA}
  ${THRUST_PARTIALLY_IMPLEMENTED_CPP}
  ${THRUST_PARTIALLY_IMPLEMENTED_TBB}
  ${THRUST_PARTIALLY_IMPLEMENTED_OMP}
)

if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  if (14 EQUAL ${CMAKE_CXX_STANDARD})
    # Temporarily disable until NVBug 2492786 is fixed.
    list(APPEND THRUST_PARTIALLY_IMPLEMENTED tuple_algorithms)
  endif()
endif ()

list(REMOVE_DUPLICATES THRUST_PARTIALLY_IMPLEMENTED)

foreach (THRUST_TEST_SOURCE IN LISTS THRUST_TESTS)
  # TODO: Per-test flags.

  set(THRUST_TEST_CREATION_ADDITIONAL)
  set(THRUST_TEST_ADD_TO_CTEST ON)

  get_filename_component(THRUST_TEST_CATEGORY ${THRUST_TEST_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_TEST_CATEGORY}"))
    set(THRUST_TEST_CATEGORY "${THRUST_TEST_CATEGORY}.")
  endif ()

  get_filename_component(THRUST_TEST_NAME ${THRUST_TEST_SOURCE} NAME_WE)

  if ("${THRUST_TEST_NAME}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED)
    # This test is partially implemented on _some_ backends...
    if (NOT "${THRUST_TEST_NAME}" IN_LIST THRUST_PARTIALLY_IMPLEMENTED_${THRUST_DEVICE_SYSTEM})
      # ...but not on the selected one.
      set(THRUST_TEST_CREATION_ADDITIONAL EXCLUDE_FROM_ALL)
      set(THRUST_TEST_ADD_TO_CTEST OFF)
    endif ()
  endif ()

  set(THRUST_TEST "thrust.test.${THRUST_TEST_CATEGORY}${THRUST_TEST_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # Test files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. But since they are pretty much just C++, we can compile them with
    # non-nvcc C++ compilers... but we need to tell CMake that they are, in fact, just C++.
    set_source_files_properties(${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
      PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
  endif ()

  add_executable(
    ${THRUST_TEST}
    ${THRUST_TEST_CREATION_ADDITIONAL}
    # THRUST_TEST_CREATION_ADDITIONAL is actually a CMake keyword (sometimes).
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  target_include_directories(
    ${THRUST_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dependencies/cub
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_link_libraries(${THRUST_TEST}
    thrust_testframework
    ${THRUST_ADDITIONAL_LIBRARIES})

  if (THRUST_TEST_ADD_TO_CTEST)
    add_test(NAME ${THRUST_TEST}
      COMMAND ${CMAKE_COMMAND}
        -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_TEST}>
        ${THRUST_TEST_RUN_ARGUMENTS})
  endif ()

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}" AND THRUST_ENABLE_TESTS_WITH_RDC)
    set(THRUST_TEST_RDC "thrust.test.${THRUST_TEST_CATEGORY}rdc.${THRUST_TEST_NAME}")

    add_executable(
      ${THRUST_TEST_RDC}
      ${THRUST_TEST_CREATION_ADDITIONAL}
      # THRUST_TEST_CREATION_ADDITIONAL is actually a CMake keyword (sometimes).
      ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
    )

    target_include_directories(
      ${THRUST_TEST_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dependencies/cub
      PRIVATE ${PROJECT_SOURCE_DIR}/testing
    )

    target_link_libraries(${THRUST_TEST_RDC}
      thrust_testframework
      ${THRUST_ADDITIONAL_LIBRARIES})

    set_target_properties(${THRUST_TEST_RDC}
      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if (THRUST_TEST_ADD_TO_CTEST)
      add_test(NAME ${THRUST_TEST_RDC}
        COMMAND ${CMAKE_COMMAND}
          -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_TEST_RDC}>
          ${THRUST_TEST_RUN_ARGUMENTS})
    endif ()
  endif ()
endforeach ()

# Handle examples.

option(THRUST_EXAMPLE_FILECHECK_PATH "Path to the LLVM FileCheck utility." "")
option(THRUST_ENABLE_EXAMPLES_WITH_RDC "Also build all examples with RDC." OFF)

set(THRUST_EXAMPLE_FILECHECK_ENABLED OFF)
if (NOT "" STREQUAL "${THRUST_EXAMPLE_FILECHECK_PATH}")
  execute_process(
    COMMAND "${THRUST_EXAMPLE_FILECHECK_PATH}" "${THRUST_FILECHECK_DATA_PATH}/thrust.sanity.filecheck"
    INPUT_FILE "${CMAKE_SOURCE_DIR}/cmake/sanity"
    RESULT_VARIABLE THRUST_FILECHECK_RESULT
  )

  if ("0" STREQUAL "${THRUST_FILECHECK_RESULT}")
    set(THRUST_EXAMPLE_FILECHECK_ENABLED ON)
    message("-- FileCheck enabled: ${THRUST_EXAMPLE_FILECHECK_PATH}")
  endif ()
endif ()

list(APPEND THRUST_EXAMPLE_GLOBS examples/*.cu)
list(APPEND THRUST_EXAMPLE_GLOBS examples/*.cpp)

if     ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_EXAMPLE_GLOBS examples/cuda/*.cu)
elseif ("OMP" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  list(APPEND THRUST_EXAMPLE_GLOBS examples/omp/*.cu)
  list(APPEND THRUST_EXAMPLE_GLOBS examples/omp/*.cpp)
endif ()

if (CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB THRUST_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${THRUST_EXAMPLE_GLOBS}
    CONFIGURE_DEPENDS
  )
else ()
  file(
    GLOB THRUST_EXAMPLES
    RELATIVE ${PROJECT_SOURCE_DIR}/examples
    ${THRUST_EXAMPLE_GLOBS}
  )
endif ()

set(THRUST_EXAMPLE_RUN_ARGUMENTS
  -DTHRUST_SOURCE=${CMAKE_SOURCE_DIR}
  -DTHRUST_FILECHECK_ENABLED=${THRUST_EXAMPLE_FILECHECK_ENABLED}
  -DTHRUST_FILECHECK=${THRUST_EXAMPLE_FILECHECK_PATH}
  -P "${CMAKE_SOURCE_DIR}/cmake/run_example.cmake")

foreach (THRUST_EXAMPLE_SOURCE IN LISTS THRUST_EXAMPLES)
  # TODO: Per-example flags.

  get_filename_component(THRUST_EXAMPLE_CATEGORY ${THRUST_EXAMPLE_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_EXAMPLE_CATEGORY}"))
    set(THRUST_EXAMPLE_CATEGORY "${THRUST_EXAMPLE_CATEGORY}.")
  endif ()

  get_filename_component(THRUST_EXAMPLE_NAME ${THRUST_EXAMPLE_SOURCE} NAME_WE)

  set(THRUST_EXAMPLE "thrust.example.${THRUST_EXAMPLE_CATEGORY}${THRUST_EXAMPLE_NAME}")

  if (NOT "CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}")
    # Example files are generally .cu; if CUDA is not enabled, CMake doesn't know what to
    # do with them. But since they are pretty much just C++, we can compile them with
    # non-nvcc C++ compilers... but we need to tell CMake that they are, in fact, just C++.
    set_source_files_properties(${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
      PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "${THRUST_TREAT_FILE_AS_CXX}")
  endif ()

  add_executable(
    ${THRUST_EXAMPLE}
    ${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
  )

  target_include_directories(
    ${THRUST_EXAMPLE}
    PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dependencies/cub
    PRIVATE ${PROJECT_SOURCE_DIR}/examples
  )

  target_link_libraries(${THRUST_EXAMPLE}
    ${THRUST_ADDITIONAL_LIBRARIES})

  add_test(NAME ${THRUST_EXAMPLE}
    COMMAND ${CMAKE_COMMAND}
      -DTHRUST_EXAMPLE=${THRUST_EXAMPLE}
      -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_EXAMPLE}>
      ${THRUST_EXAMPLE_RUN_ARGUMENTS})

  if ("CUDA" STREQUAL "${THRUST_DEVICE_SYSTEM}" AND THRUST_ENABLE_EXAMPLES_WITH_RDC)
    set(THRUST_EXAMPLE_RDC "thrust.example.${THRUST_EXAMPLE_CATEGORY}rdc.${THRUST_EXAMPLE_NAME}")

    add_executable(
      ${THRUST_EXAMPLE_RDC}
      ${PROJECT_SOURCE_DIR}/examples/${THRUST_EXAMPLE_SOURCE}
    )

    target_include_directories(
      ${THRUST_EXAMPLE_RDC}
      PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/dependencies/cub
      PRIVATE ${PROJECT_SOURCE_DIR}/examples
    )

    target_link_libraries(${THRUST_EXAMPLE_RDC}
      ${THRUST_ADDITIONAL_LIBRARIES})

    set_target_properties(${THRUST_EXAMPLE_RDC}
      PROPERTIES CUDA_SEPERABLE_COMPILATION ON)

    add_test(NAME ${THRUST_EXAMPLE_RDC}
      COMMAND ${CMAKE_COMMAND}
        -DTHRUST_EXAMPLE=${THRUST_EXAMPLE}
        -DTHRUST_BINARY=$<TARGET_FILE:${THRUST_EXAMPLE_RDC}>
        ${THRUST_EXAMPLE_RUN_ARGUMENTS})
  endif ()
endforeach ()

