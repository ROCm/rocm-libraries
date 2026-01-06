# ########################################################################
# Copyright (C) 2018-2020 Advanced Micro Devices, Inc. All rights Reserved.
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

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

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

# This project may compile dependencies for clients
project(hipsparse-clients LANGUAGES CXX)

if (BUILD_CLIENTS_ONLY)
  find_package(hipsparse 4.0.0 REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT TARGET hipsparse)
  find_package(hipsparse REQUIRED CONFIG PATHS /opt/rocm/hipsparse)

  option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
  option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
  option(BUILD_CLIENTS_BENCHMARKS "Build benchmarks" ON)
endif()

# Set CXX flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CXX standard
if (CMAKE_CXX_STANDARD EQUAL 14)
  message( DEPRECATION "Builds using the C++14 standard will no longer be supported in the next major release" )
elseif(NOT CMAKE_CXX_STANDARD EQUAL 17)
  message(FATAL_ERROR "Only C++14 and C++17 are supported")
endif()

# If OpenMP is available, we can use it to speed up some tests
find_package(OpenMP QUIET)
find_package(Threads QUIET)

if(OPENMP_FOUND AND THREADS_FOUND)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    # OpenMP cmake fix for cmake <= 3.9
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(tests)

  if(NOT CONVERT_SOURCE)
    set(CONVERT_SOURCE ${CMAKE_SOURCE_DIR}/deps/convert.cpp)
  endif()
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${CONVERT_SOURCE} -Wl,--build-id=sha1 -O3 -o ${PROJECT_BINARY_DIR}/mtx2csr.exe RESULT_VARIABLE STATUS)
  if(STATUS AND NOT STATUS EQUAL 0)
    message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.")
  endif()

  set(HIPSPARSE_CLIENTMATRICES "${CMAKE_SOURCE_DIR}/cmake/hipsparse_clientmatrices.cmake")

  set(HIPSPARSE_CONVERT "${PROJECT_BINARY_DIR}/hipsparse_mtx2csr")
  add_custom_command(OUTPUT "${HIPSPARSE_CONVERT}"
                    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/mtx2csr.exe" "${HIPSPARSE_CONVERT}"
                    DEPENDS "${PROJECT_BINARY_DIR}/mtx2csr.exe"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(HIPSPARSE_COMMON "${PROJECT_BINARY_DIR}/staging/hipsparse_common.yaml")
  add_custom_command(OUTPUT "${HIPSPARSE_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipsparse_common.yaml "${HIPSPARSE_COMMON}"
                    DEPENDS include/hipsparse_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(HIPSPARSE_TEMPLATE "${PROJECT_BINARY_DIR}/staging/hipsparse_template.yaml")
  add_custom_command(OUTPUT "${HIPSPARSE_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/hipsparse_template.yaml "${HIPSPARSE_TEMPLATE}"
                    DEPENDS include/hipsparse_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  set(HIPSPARSE_GENTEST "${PROJECT_BINARY_DIR}/staging/hipsparse_gentest.py")
  add_custom_command(OUTPUT "${HIPSPARSE_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/hipsparse_gentest.py "${HIPSPARSE_GENTEST}"
                    DEPENDS common/hipsparse_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

  add_custom_target(hipsparse-common DEPENDS "${HIPSPARSE_COMMON}" "${HIPSPARSE_TEMPLATE}" "${HIPSPARSE_CLIENTMATRICES}" "${HIPSPARSE_GENTEST}" "${HIPSPARSE_CONVERT}")

  rocm_install(
    FILES "${HIPSPARSE_COMMON}" "${HIPSPARSE_TEMPLATE}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test"
    )
  #
  rocm_install(
    PROGRAMS "${HIPSPARSE_GENTEST}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/hipsparse/test"
    )

  rocm_install(
    PROGRAMS "${HIPSPARSE_CONVERT}"
    COMPONENT clients-common
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  rocm_install(
    FILES "${HIPSPARSE_CLIENTMATRICES}"
    COMPONENT clients-common
    DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test"
    )

endif()
