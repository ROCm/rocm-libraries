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
project(hipsolver-clients LANGUAGES CXX)
if(UNIX)
  enable_language(Fortran)
endif()

# We use C++17 features, this will add compile option: -std=c++17
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(build-options)

# Linking lapack library requires fortran flags
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
if(HIPSOLVER_FIND_PACKAGE_LAPACK_CONFIG)
  find_package(LAPACK 3.7 REQUIRED CONFIG)
else()
  find_package(LAPACK 3.7 REQUIRED)
endif()

if(NOT LAPACK_LIBRARIES)
  set(LAPACK_LIBRARIES
    ${LAPACK_blas_LIBRARIES}
    ${LAPACK_lapack_LIBRARIES}
  )
endif()

if(NOT TARGET hipsolver)
  find_package(hipsolver REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)
endif()

if(BUILD_FORTRAN_BINDINGS)
  set(hipsolver_f90_source_clients
    include/hipsolver_fortran.f90
  )
endif()

if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS)
  if(BUILD_FORTRAN_BINDINGS)
    add_library(hipsolver_fortran_client STATIC ${hipsolver_f90_source_clients})
    target_link_libraries(hipsolver_fortran_client hipsolver_fortran)
    include_directories(${CMAKE_BINARY_DIR}/include/hipsolver)
    include_directories(${CMAKE_BINARY_DIR}/include/hipsolver/internal)
    target_compile_definitions(hipsolver_fortran_client INTERFACE HAVE_HIPSOLVER_FORTRAN_CLIENT)
  endif()

  add_library(clients-common INTERFACE)
  target_include_directories(clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(clients-common INTERFACE
    $<$<PLATFORM_ID:Linux>:stdc++fs>
  )
  set(common_source_files
    common/clients_utility.cpp
    common/hipsolver_datatype2string.cpp
    common/lapack_host_reference.cpp
    common/utility.cpp
    rocsolvercommon/rocsolver_test.cpp
  )

  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)
  target_sources(clients-common INTERFACE ${common_source_paths})

  # Copy and point to sparse test data
  file(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${PROJECT_BINARY_DIR}/staging/sparsedata/
  )
  install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/sparsedata/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/hipsolver/test
    COMPONENT tests
  )

  if(BUILD_CLIENTS_TESTS)
    add_subdirectory(gtest)
  endif()

  if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()
