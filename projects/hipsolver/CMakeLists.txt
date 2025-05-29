# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

cmake_minimum_required(VERSION 3.12)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
  set(CMAKE_Fortran_COMPILER  "gfortran")
endif()

project(hipsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
find_package(ROCM 0.7.3 CONFIG QUIET PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  set(rocm_cmake_url "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip")
  set(rocm_cmake_path "${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}")
  set(rocm_cmake_archive "${rocm_cmake_path}.zip")
  file(DOWNLOAD "${rocm_cmake_url}" "${rocm_cmake_archive}" STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(status_code EQUAL 0)
    message(STATUS "downloading... done")
  else()
    message(FATAL_ERROR "error: downloading\n'${rocm_cmake_url}' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}\n")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf "${rocm_cmake_archive}"
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag})
  execute_process(COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package(ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake)
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMClients)
include(ROCMHeaderWrapper)

set(VERSION_STRING "3.0.0")
rocm_setup_version(VERSION ${VERSION_STRING})

if(NOT DEFINED ENV{HIP_PATH})
  set(HIP_PATH "/opt/rocm/hip")
else()
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_PATH)
endif()

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake/hip /opt/rocm /opt/rocm/llvm /opt/rocm/hip)

option(BUILD_SHARED_LIBS "Build hipSOLVER as a shared library" ON)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_CODE_COVERAGE "Build hipSOLVER with code coverage enabled" OFF)
option(BUILD_HIPBLAS_TESTS "Build additional tests to ensure hipBLAS and hipSOLVER are compatible (requires installed hipBLAS)" OFF)
option(BUILD_HIPSPARSE_TESTS "Build additional tests to cover sparse functionality (requires installed hipSPARSE)" ON)
# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option(BUILD_WITH_SPARSE "Build hipSOLVER with sparse functionality available at build time (requires installed dependencies)" OFF)
option(BUILD_VERBOSE "Output additional build information" OFF)
option(USE_CUDA "Look for CUDA and use that as a backend if found" OFF)
option(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG "Skip module mode search for LAPACK" ON)
option(BUILD_FORTRAN_BINDINGS "Build the Fortran bindings" "${UNIX}")

if(NOT BUILD_SHARED_LIBS)
  add_compile_definitions(HIPSOLVER_STATIC_LIB)
endif()

if(BUILD_FORTRAN_BINDINGS)
  enable_language(Fortran)
endif()

add_library(hipsolver-common INTERFACE)
target_compile_options(hipsolver-common INTERFACE
  -Wno-unused-result # TODO: address [[nodiscard]] warnings
)
if(BUILD_ADDRESS_SANITIZER)
  target_compile_options(hipsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
  )
  target_link_options(hipsolver-common INTERFACE
    -fsanitize=address
    -shared-libasan
    -fuse-ld=lld
  )
endif()



# Find CUDA if the user wants a CUDA version.
if(USE_CUDA)
  find_package(CUDA REQUIRED)
endif()

# Hip headers required of all clients; clients use hip to allocate device memory
if(USE_CUDA)
  # Append cmake path for hip (for convenience)
  # Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
  list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip ${HIP_PATH}/cmake)
  find_package(HIP MODULE REQUIRED)
else()
  find_package(hip REQUIRED CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)
endif()

if(USE_CUDA)
  list(APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
endif()

# Append our library helper cmake path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEFAULT_ARMOR_LEVEL 1)
else()
  set(DEFAULT_ARMOR_LEVEL 0)
endif()
set(ARMOR_LEVEL "${DEFAULT_ARMOR_LEVEL}" CACHE STRING "Enables increasingly expensive runtime correctness checks")
include(armor-config)

# FOR OPTIONAL CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  add_compile_options(-fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

include(CMakeDependentOption)
cmake_dependent_option(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY
  "Build with file/folder reorg backward compatibility enabled" OFF "NOT WIN32" OFF)
if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
endif()

if(WIN32)
  add_compile_definitions(
    WIN32_LEAN_AND_MEAN
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
  )
endif()

add_subdirectory(library)

include(clients/cmake/build-options.cmake)

if(NOT SYSTEM_OS)
  rocm_set_os_id(SYSTEM_OS)
  string(TOLOWER "${SYSTEM_OS}" SYSTEM_OS)
  rocm_read_os_release(SYSTEM_OS_VERSION VERSION_ID)
endif()

# Build clients of the library
if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS)
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(SYSTEM_OS STREQUAL "centos" OR SYSTEM_OS STREQUAL "rhel" OR SYSTEM_OS STREQUAL "almalinux")
    if(SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(SYSTEM_OS STREQUAL "ubuntu" AND SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(SYSTEM_OS STREQUAL "mariner" OR SYSTEM_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()
  rocm_package_setup_component(clients)
  if(UNIX)
    set(DEP_ARGS DEPENDS RPM "${GFORTRAN_RPM}" DEB "${GFORTRAN_DEB}")
  endif()
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(tests ${DEP_ARGS})
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(benchmarks ${DEP_ARGS})
  endif()
  add_subdirectory(clients)
endif()

# Package specific CPACK vars
if(NOT USE_CUDA)
  set(rocblas_minimum 4.2.0)
  set(rocsolver_minimum 3.30.0)
  rocm_package_add_dependencies(SHARED_DEPENDS "rocblas >= ${rocblas_minimum}" "rocsolver >= ${rocsolver_minimum}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocblas-static-devel >= ${rocblas_minimum}" "rocsolver-static-devel >= ${rocsolver_minimum}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocblas-static-dev >= ${rocblas_minimum}" "rocsolver-static-dev >= ${rocsolver_minimum}")

  if(SYSTEM_OS STREQUAL "centos" OR SYSTEM_OS STREQUAL "rhel" OR SYSTEM_OS STREQUAL "mariner" OR SYSTEM_OS STREQUAL "azurelinux")
    list(APPEND hipsolver_pkgdeps "suitesparse")
  elseif(SYSTEM_OS STREQUAL "ubuntu" AND SYSTEM_OS_VERSION VERSION_GREATER_EQUAL "24.04")
    list(APPEND hipsolver_pkgdeps "libcholmod5" "libsuitesparseconfig7")
  else()
    list(APPEND hipsolver_pkgdeps "libcholmod3" "libsuitesparseconfig5")
  endif()

  if(BUILD_WITH_SPARSE)
    set(rocsparse_minimum 2.3.0)
    rocm_package_add_dependencies(SHARED_DEPENDS "rocsparse >= ${rocsparse_minimum}")
    rocm_package_add_rpm_dependencies(STATIC_DEPENDS "rocsparse-static-devel >= ${rocsparse_minimum}")
    rocm_package_add_deb_dependencies(STATIC_DEPENDS "rocsparse-static-dev >= ${rocsparse_minimum}")

    rocm_package_add_dependencies(DEPENDS ${hipsolver_pkgdeps})
  else()
    string(JOIN "," CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "rocsparse" ${hipsolver_pkgdeps})
    string(JOIN "," CPACK_RPM_RUNTIME_PACKAGE_SUGGESTS "rocsparse" ${hipsolver_pkgdeps})
  endif()
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

if(WIN32)
  set(CPACK_SOURCE_GENERATOR "ZIP")
  set(CPACK_GENERATOR "ZIP")
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
    set(CPACK_PACKAGING_INSTALL_PREFIX "${ROCM_PATH}")
  endif()
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}")

# Give hipsolver compiled for CUDA backend a different name
if(NOT USE_CUDA)
  set(package_name hipsolver)
else()
  set(package_name hipsolver-alt)
endif()

set(HIPSOLVER_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm LAPACK marshalling library"
    MAINTAINER "hipSOLVER Maintainer <hipsolver-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${HIPSOLVER_CONFIG_DIR}
)

# ADDITIONAL TARGETS FOR CODE COVERAGE
#
# > make coverage_cleanup (clean coverage related files.)
# > make coverage GTEST_FILTER=<>
# will run:
#  > make coverage_analysis GTEST_FILTER=<> (analyze tests)
#  > make coverage_output (generate html documentation)
if(BUILD_CODE_COVERAGE)
  set(coverage_test ./clients/staging/hipsolver-test)

  # Run coverage analysis
  add_custom_target(coverage_analysis
     COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
     COMMAND ${coverage_test} --gtest_filter=\"\${GTEST_FILTER}\"
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  add_dependencies(coverage_analysis hipsolver)

  # Prepare coverage output
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  add_custom_target(coverage_output
      DEPENDS coverage_analysis
      COMMAND mkdir -p lcoverage
      COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
      COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
      COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
      COMMAND chmod +x llvm-gcov.sh
  )

  # Generate coverage output.
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml --ignore-errors source lcoverage/main_coverage.info --output-directory lcoverage
  )

  add_custom_target(coverage DEPENDS coverage_output)

  # Coverage cleanup
  add_custom_target(coverage_cleanup
      COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
