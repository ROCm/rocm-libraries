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

add_definitions(-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_${THRUST_HOST_SYSTEM})

set(THRUST_DEVICE_SYSTEM_OPTIONS CUDA CPP OMP TBB)
set(THRUST_DEVICE_SYSTEM CUDA CACHE STRING "The device backend to target.")
set_property(
  CACHE THRUST_DEVICE_SYSTEM
  PROPERTY STRINGS ${THRUST_DEVICE_SYSTEM_OPTIONS}
)

if (NOT THRUST_DEVICE_SYSTEM IN_LIST THRUST_DEVICE_SYSTEM_OPTIONS)
  message(
    FATAL_ERROR
    "THRUST_DEVICE_SYSTEM must be one of ${THRUST_DEVICE_SYSTEM_OPTIONS}"
  )
endif ()

if ("OMP" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif ()

if ("TBB" STREQUAL "${THRUST_DEVICE_SYSTEM}")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(TBB tbb REQUIRED)
  if (TBB_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TDD_CFLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TDD_CFLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TBB_LD_FLAGS}")
    set (THRUST_ADDITIONAL_LIBRARIES "${TBB_LIBRARIES}")
  endif ()

  # There's a ton of these in the TBB backend, even though the code is correct.
  # TODO: silence these warnings in code instead
  append_option_if_available("-Wno-unused-parameter" THRUST_CXX_WARNINGS)
endif ()

if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 1900)
    message(FATAL_ERROR "This version of MSVC no longer supported.")
  endif ()
endif ()

# Set CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(DISABLE_WERROR)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-function")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-function -Werror")
endif()

# Get dependencies
include(cmake/Dependencies.cmake)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")

  # Disable warning about applying unary operator- to unsigned type.
  append_option_if_available(CXX "/wd4146" THRUST_OPTIONS_WARNINGS)
else ()
  append_option_if_available(CXX "-Werror" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wall" THRUST_O:TIONS_WARNINGS)
  append_option_if_available(CXX "-Wextra" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Winit-self" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Woverloaded-virtual" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wcast-qual" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-cast-align" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-long-long" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-variadic-macros" THRUST_OPTIONS_WARNINGS)
  append_option_if_available(CXX "-Wno-unused-function" THRUST_OPTIONS_WARNINGS)
endif ()

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

if (("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}") OR
    ("XL" STREQUAL "${CMAKE_CXX_COMPILER_ID}"))
  # xlC and Clang warn about unused parameters in uninstantiated templates.
  # This causes xlC to choke on the OMP backend, which is mostly #ifdef'd out
  # (and thus has unused parameters) when you aren't using it.
  append_option_if_available("-Wno-unused-parameters" THRUST_CXX_WARNINGS)
endif ()
        
if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # -Wunneeded-internal-declaration misfires in the unit test framework
  # on older versions of Clang.
  append_option_if_available("-Wno-unneeded-internal-declaration" THRUST_CXX_WARNINGS)
endif ()
  
append_option_if_available(CUDA "-rdc=true" THRUST_OPTIONS_RDC)

set(THRUST_OPTIONS_DEBUG ${THRUST_OPTIONS_WARNINGS})
set(THRUST_OPTIONS_RELEASE ${THRUST_OPTIONS_WARNINGS})

# Package
set(CPACK_DEBIAN_ARCHIVE_TYPE "gnutar")

list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/testframework.cu)
if ("CUDA" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TESTFRAMEWORK_FILES testing/unittest/cuda/testframework.cu)
endif ()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

if     ("CUDA" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TEST_GLOBS testing/cuda/*.cu)
elseif ("OMP" STREQUAL "${THRUST_DEVICE_BACKEND}")
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cu)
  list(APPEND THRUST_TEST_GLOBS testing/omp/*.cpp)
endif ()

if (CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB THRUST_TESTS
    RELATIVE ${PROJECT_SOURCE_DIR}/testing
    ${THRUST_TEST_GLOBS}
    CONFIGURE_DEPENDS
  ) 
else ()
  file(
    GLOB THRUST_TESTS
    RELATIVE ${PROJECT_SOURCE_DIR}/testing
    ${THRUST_TEST_GLOBS}
  ) 
endif ()

foreach (THRUST_TEST_SOURCE IN LISTS THRUST_TESTS)
  # TODO: Per-test flags.

  get_filename_component(THRUST_TEST_CATEGORY ${THRUST_TEST_SOURCE} DIRECTORY)
  if (NOT ("" STREQUAL "${THRUST_TEST_CATEGORY}"))
    set(THRUST_TEST_CATEGORY "${THRUST_TEST_CATEGORY}.")
  endif () 

  get_filename_component(THRUST_TEST ${THRUST_TEST_SOURCE} NAME_WE)

  set(THRUST_TEST "thrust.test.${THRUST_TEST_CATEGORY}${THRUST_TEST}")
  set(THRUST_TEST_RDC "thrust.test.${THRUST_TEST_CATEGORY}rdc.${THRUST_TEST}")

  add_executable(
    ${THRUST_TEST}
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  add_executable(
    ${THRUST_TEST_RDC}
    ${PROJECT_SOURCE_DIR}/testing/${THRUST_TEST_SOURCE}
  )

  target_compile_options(${THRUST_TEST}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE}>")

  target_compile_options(${THRUST_TEST_RDC}
    PRIVATE "$<$<CONFIG:DEBUG>:${THRUST_OPTIONS_DEBUG} ${THRUST_OPTIONS_RDC}>"
            "$<$<CONFIG:RELEASE>:${THRUST_OPTIONS_RELEASE} ${THRUST_OPTIONS_RDC}>")

  target_include_directories(
    ${THRUST_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_include_directories(
    ${THRUST_TEST_RDC}
    PUBLIC ${PROJECT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/testing
  )

  target_link_libraries(${THRUST_TEST}     thrust_testframework)
  target_link_libraries(${THRUST_TEST_RDC} thrust_testframework)

  add_test(run.${THRUST_TEST}     ${THRUST_TEST})
  add_test(run.${THRUST_TEST_RDC} ${THRUST_TEST_RDC})
endforeach ()

