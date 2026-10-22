# ########################################################################
# Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX ${ROCM_PATH} CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# Needed to pickup static library files
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH})

# CMake modules
list(APPEND CMAKE_MODULE_PATH
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake
     ${ROCM_PATH}/lib/cmake/hip
     ${ROCM_PATH}/hip/cmake
     ${ROCM_PATH}/cmake)

# Root of the rocm-libraries monorepo (for shared infrastructure like CTest categories)
if(NOT DEFINED ROCM_LIBRARIES_ROOT)
  set(ROCM_LIBRARIES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Honor per-config flags in try_compile() source-file signature. cmake v3.7 and up
if(POLICY CMP0066)
  cmake_policy(SET CMP0066 NEW)
endif()

# rocALUTION project
project(rocalution LANGUAGES CXX)

# Build flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

# Build options
option(BUILD_SHARED_LIBS "Build rocALUTION as a shared library" ON)
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks (requires boost)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
option(BUILD_CLIENTS_ONLY "Build only clients" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(BUILD_CODE_COVERAGE "Build with code coverage enabled" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_GLOBALTYPE_64 "Support global number of rows / columns exceeding 32 bits" OFF)
option(BUILD_LOCALTYPE_64 "Support local number of rows / columns exceeding 32 bits" OFF)
option(BUILD_PTRTYPE_64 "Support local number of non-zeros exceeding 32 bits" OFF)
option(BUILD_OPTCPU "Enable all instruction subsets supported by the local machine" OFF)
option(BUILD_SUPPORT_COMPLEX "Enable complex number support" ON)

# Dependencies
include(cmake/Dependencies.cmake)

if(BUILD_ADDRESS_SANITIZER)
  set(DEFAULT_AMDGPU_TARGETS
    amdgcnspirv
    gfx908:xnack+
    gfx90a:xnack+
    gfx942:xnack+
  )
else()
  set(DEFAULT_AMDGPU_TARGETS
    amdgcnspirv
    gfx803
    gfx900:xnack-
    gfx906:xnack-
    gfx908:xnack-
    gfx90a:xnack-
    gfx90a:xnack+
    gfx942
    gfx950
    gfx1030
    gfx1100
    gfx1101
    gfx1102
    gfx1150
    gfx1151
    gfx1152
    gfx1153
    gfx1200
    gfx1201
    gfx1250-strict
  )
endif()
if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
  message( DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS." )
endif()
set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "Target default GPUs if AMDGPU_TARGETS is not defined. (Deprecated, prefer GPU_TARGETS)")
# Don't force, users should be able to override GPU_TARGETS at the command line if desired
set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to build for")

include(CheckLanguage)
include(CMakeDependentOption)
check_language(HIP)
cmake_dependent_option(USE_HIPCXX "Use CMake HIP language support" OFF CMAKE_HIP_COMPILER OFF)
if(USE_HIPCXX)
  enable_language(HIP)
else()
  find_package(HIP MODULE) # hip_add_library is only provided by the find module
  if(NOT HIP_FOUND)
    message("-- HIP not found. Compiling WITHOUT HIP support.")
  endif()
endif()

cmake_dependent_option(SUPPORT_HIP "Compile WITH HIP support" ON "USE_HIPCXX OR HIP_FOUND" OFF)

# HIP related library dependencies
if(SUPPORT_HIP)
  if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
  endif( )
  find_package(rocblas REQUIRED)
  find_package(rocsparse REQUIRED)
  find_package(rocprim REQUIRED)
  find_package(rocrand REQUIRED)
endif()

if(BUILD_CODE_COVERAGE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "BUILD_CODE_COVERAGE requires a clang compiler, e.g. amdclang++")
  endif()

  # Locate the host profile runtime so coverage targets can link it directly.
  # Passing -fprofile-instr-generate to a HIP link instead puts the device
  # profile runtime into the device image, and the HIP runtime then aborts while
  # dumping the profile at process exit (AISPARSE-778).
  #
  # The profile runtime is not self-contained: its ROCm platform object pulls in
  # the device-drain and symbol-interception helpers, so those archives have to
  # be named too. Link them as a group, since they reference each other.
  set(ROCALUTION_PROFILE_RUNTIME)
  foreach(_rt clang_rt.profile clang_rt.profile_rocm
              clang_rt.interception clang_rt.sanitizer_common)
    foreach(_suffix "-x86_64" "")
      execute_process(COMMAND ${CMAKE_CXX_COMPILER}
                              --print-file-name=lib${_rt}${_suffix}.a
                      OUTPUT_VARIABLE _rt_path OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(EXISTS "${_rt_path}")
        list(APPEND ROCALUTION_PROFILE_RUNTIME "${_rt_path}")
        break()
      endif()
    endforeach()
  endforeach()
  if(NOT ROCALUTION_PROFILE_RUNTIME)
    message(FATAL_ERROR
      "BUILD_CODE_COVERAGE: could not locate the clang profile runtime")
  endif()
  message(STATUS "Host profile runtime: ${ROCALUTION_PROFILE_RUNTIME}")
  set(ROCALUTION_PROFILE_RUNTIME
      -Wl,--start-group ${ROCALUTION_PROFILE_RUNTIME} -Wl,--end-group
      ${CMAKE_DL_LIBS} -lpthread -lrt)
endif()

if(BUILD_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
    add_link_options(-fuse-ld=lld)
endif()


# Setup version
rocm_setup_version(VERSION "4.1.1")

# Configure Windows version resource file
if(WIN32)
    string(TIMESTAMP ROCALUTION_COPYRIGHT_YEAR "%Y")
    if(rocalution_VERSION_TWEAK)
        set(ROCALUTION_PRODUCT_VERSION_DISPLAY "${rocalution_VERSION_MAJOR}.${rocalution_VERSION_MINOR} (${rocalution_VERSION_TWEAK})")
    else()
        set(ROCALUTION_PRODUCT_VERSION_DISPLAY "${rocalution_VERSION_MAJOR}.${rocalution_VERSION_MINOR}")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/version.rc.in"
                   "${PROJECT_BINARY_DIR}/version.rc" @ONLY)
endif()

if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  if(BUILD_CLIENTS_TESTS)
    enable_testing()
    rocm_package_setup_client_component(tests)
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks)
  endif()
endif()

# rocALUTION library
if (NOT BUILD_CLIENTS_ONLY)
  add_subdirectory(src)
endif()

# Trigger client builds if selected
if(BUILD_CLIENTS_ONLY OR BUILD_CLIENTS)
  add_subdirectory(clients)
endif()

# Package specific CPACK vars
if(SUPPORT_HIP)
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan >= 4.5.0" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd >= 4.5.0" )
  endif()
  rocm_package_add_dependencies(DEPENDS "${DEPENDS_HIP_RUNTIME}" "rocsparse >= 1.12.10" "rocblas >= 2.22.0" "rocrand >= 2.1.0")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
  # Only force the default Windows SDK install path when the caller did not set
  # one explicitly.
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
  endif()
  set(INSTALL_PREFIX "C:/hipSDK")
  set(CPACK_SET_DESTDIR OFF)
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
  set(CPACK_PACKAGING_INSTALL_PREFIX "")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
  if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# Package name
set(package_name rocalution)

set(ROCALUTION_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "ROCm library for sparse linear systems"
  MAINTAINER "rocALUTION Maintainer <rocalution-maintainer@amd.com>"
  LDCONFIG
  LDCONFIG_DIR ${ROCALUTION_CONFIG_DIR}
)

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
#
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  # > make coverage GTEST_FILTER=<>
  # will run:
  #  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
  # and then write the lcov and html reports to coverage-report
  #

  #
  # Run coverage analysis
  #
  add_custom_target(coverage_analysis
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/rocalution-coverage_%m.profraw" ./clients/staging/rocalution-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running rocalution coverage analysis"
    )

  add_dependencies(coverage_analysis rocalution)

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  # Each instrumented library writes its own profile (%m), and llvm-cov needs
  # every one of them as an object to attribute the counters
  set(COVERAGE_OBJECTS -object $<TARGET_FILE:rocalution>)
  if(TARGET rocalution_hip)
    list(APPEND COVERAGE_OBJECTS -object $<TARGET_FILE:rocalution_hip>)
  endif()

  # llvm-cov does not process the LCOV_EXCL_START/LCOV_EXCL_STOP markers. The
  # exported report is filtered for those, and for sources that are not part of
  # the library, e.g. ROCm headers or the logging utilities
  set(COVERAGE_FILTER_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/filter_lcov_exclusions.py")
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # Multi-node code can only run in a build with MPI, so it is excluded otherwise
  set(COVERAGE_FILTER_OPTIONS)
  if(NOT SUPPORT_MPI)
    list(APPEND COVERAGE_FILTER_OPTIONS --no-mpi)
  endif()
  find_program(GENHTML genhtml)

  if(GENHTML)
    set(COVERAGE_HTML_COMMAND
      COMMAND ${GENHTML} ./coverage-report/coverage.info --output-directory ./coverage-report --title rocalution --legend --demangle-cpp --quiet
    )
  else()
    message(STATUS "genhtml not found - using llvm-cov show (exclusion markers not reflected in HTML)")
    set(COVERAGE_HTML_COMMAND
      COMMAND ${LLVM_COV} show ${COVERAGE_OBJECTS} -instr-profile=./coverage-report/rocalution.profdata -format=html -output-dir=coverage-report
    )
  endif()

  add_custom_target(
    coverage
    DEPENDS coverage_analysis
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocalution-coverage_*.profraw -o ./coverage-report/rocalution.profdata
    COMMAND ${LLVM_COV} report ${COVERAGE_OBJECTS} -instr-profile=./coverage-report/rocalution.profdata
    COMMAND ${LLVM_COV} export ${COVERAGE_OBJECTS} -instr-profile=./coverage-report/rocalution.profdata -format=lcov > ./coverage-report/coverage-raw.info
    COMMAND ${Python3_EXECUTABLE} ${COVERAGE_FILTER_SCRIPT} ${COVERAGE_FILTER_OPTIONS} ./coverage-report/coverage-raw.info ./coverage-report/coverage.info
    ${COVERAGE_HTML_COMMAND}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Building coverage report"
  )

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND ${CMAKE_COMMAND} -E rm -rf ./coverage-report
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning up coverage data files"
    )
endif()
