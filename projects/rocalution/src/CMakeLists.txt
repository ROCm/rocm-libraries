# ########################################################################
# Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the rocALUTION version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/base/version.hpp.in"
               "${PROJECT_BINARY_DIR}/include/rocalution/version.hpp"
)

# Default integer types
if(BUILD_GLOBALTYPE_64)
  set(rocalution_GLOBAL_TYPE "int64_t")
else()
  set(rocalution_GLOBAL_TYPE "int32_t")
endif()

if(BUILD_LOCALTYPE_64)
  set(rocalution_LOCAL_TYPE "int64_t")
else()
  set(rocalution_LOCAL_TYPE "int32_t")
endif()

if(BUILD_PTRTYPE_64)
  set(rocalution_PTR_TYPE "int64_t")
else()
  set(rocalution_PTR_TYPE "int32_t")
endif()

# Configure a header file to specify rocALUTION indexing types
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/types.hpp.in"
               "${PROJECT_BINARY_DIR}/include/rocalution/utils/types.hpp"
)

# Include sub-directories
include(base/CMakeLists.txt)
include(base/host/CMakeLists.txt)
include(solvers/CMakeLists.txt)
include(utils/CMakeLists.txt)

if(SUPPORT_HIP)
  include(base/hip/CMakeLists.txt)
endif()

# Public rocALUTION headers
set(PUBLIC_HEADERS
    rocalution.hpp
    ${BASE_PUBLIC_HEADERS}
    ${SOLVERS_PUBLIC_HEADERS}
    ${UTILS_PUBLIC_HEADERS}
)

# Copy public headers to include directory
foreach(i ${PUBLIC_HEADERS})
  configure_file("${i}" "${PROJECT_BINARY_DIR}/include/rocalution/${i}" COPYONLY)
endforeach()

source_group("Header Files\\Public" FILES ${PUBLIC_HEADERS})

# rocALUTION source
set(SOURCE
    ${BASE_SOURCES}
    ${HOST_SOURCES}
    ${SOLVERS_SOURCES}
    ${UTILS_SOURCES}
)

if(SUPPORT_MPI)
  list(APPEND SOURCE ${UTILS_MPI_SOURCES})
endif()

# Create rocALUTION host library
add_library(rocalution ${SOURCE} ${PUBLIC_HEADERS})
add_library(roc::rocalution ALIAS rocalution)

set(package_depends)
set(static_depends)

# Target link libraries
if(SUPPORT_OMP)
  if(CMAKE_VERSION VERSION_LESS "3.29.0"
      AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      AND (CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      AND (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))
    # Versions of CMake before 3.29.0 didn't correctly add the OpenMP library
    # to the OpenMP::OpenMP_CXX target when using LLVM Clang with GNU-compatible
    # front-end (i.e., `clang++`, not `clang-cl`) targeting the MSVC ABI.
    # Work around that issue by manually linking to libomp.
    target_link_libraries(rocalution PRIVATE OpenMP::OpenMP_CXX libomp)
  else()
    target_link_libraries(rocalution PRIVATE OpenMP::OpenMP_CXX)
  endif()
  list(APPEND package_depends PACKAGE OpenMP)
endif()

if(SUPPORT_MPI)
  target_link_libraries(rocalution PUBLIC MPI::MPI_CXX)
  list(APPEND static_depends PACKAGE MPI)
endif()

# Target compile definitions
if(SUPPORT_MPI)
  target_compile_definitions(rocalution PRIVATE SUPPORT_MULTINODE)
endif()

if(BUILD_SUPPORT_COMPLEX)
  target_compile_definitions(rocalution PRIVATE SUPPORT_COMPLEX)
endif()

if(SUPPORT_HIP)
  target_compile_definitions(rocalution PRIVATE SUPPORT_HIP)
  list(APPEND package_depends PACKAGE HIP)
endif()

# Target properties
rocm_set_soversion(rocalution "1.0")
set_target_properties(rocalution PROPERTIES DEBUG_POSTFIX "-d")
if(WIN32)
  set_target_properties(rocalution PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocalution EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocalution/export.hpp)

# If CPU optimizations for local machine is actived
if(BUILD_OPTCPU)
  list(APPEND CMAKE_HOST_FLAGS "-march=native")
endif()

# Set compile options
list(APPEND CMAKE_HOST_FLAGS "-O3")
target_compile_options(rocalution PRIVATE ${CMAKE_HOST_FLAGS})

# Create rocALUTION hip library
if(SUPPORT_HIP)
  # Flag source file as a hip source file
  if(USE_HIPCXX)
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
  else()
    set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
  endif()

  if(USE_HIPCXX)
    add_library(rocalution_hip ${HIP_SOURCES})
    # If CPU optimizations for local machine is actived
    if(BUILD_OPTCPU)
      target_compile_options(rocalution_hip PRIVATE "-march=native")
    endif()
  else()
    # HIP flags workaround while target_compile_options do not work

    # If CPU optimizations for local machine is actived
    if(BUILD_OPTCPU)
      list(APPEND HIP_HIPCC_FLAGS "-march=native")
    endif()

    list(APPEND HIP_HIPCC_FLAGS "-O3 -fPIC -std=c++17")
    # Use -DNDEBUG compile option for Release and RelWithDebInfo build
    if((${CMAKE_BUILD_TYPE} MATCHES "Release") OR (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo"))
      list(APPEND HIP_HIPCC_FLAGS "-DNDEBUG")
    endif()
    foreach(target ${GPU_TARGETS})
      list(APPEND HIP_HIPCC_FLAGS "--offload-arch=${target}")
    endforeach()

    # Create rocALUTION HIP library
    if(WIN32)
      add_library(rocalution_hip OBJECT ${HIP_SOURCES})
    else()
      hip_add_library(rocalution_hip ${HIP_SOURCES})
    endif()
  endif()

  if(SUPPORT_MPI)
    target_compile_definitions(rocalution_hip PRIVATE SUPPORT_MULTINODE)
    target_link_libraries(rocalution_hip PUBLIC MPI::MPI_CXX)
  endif()

  if(BUILD_SUPPORT_COMPLEX)
    target_compile_definitions(rocalution_hip PRIVATE SUPPORT_COMPLEX)
  endif()

  rocm_set_soversion(rocalution_hip "1.0.0")
  set_target_properties(rocalution_hip PROPERTIES DEBUG_POSTFIX "-d")

  if(WIN32)
    target_link_libraries(rocalution_hip PRIVATE roc::rocblas roc::rocsparse roc::rocprim roc::rocrand hip::device)
  else()
    target_link_libraries(rocalution_hip PRIVATE roc::rocblas roc::rocsparse roc::rocprim roc::rocrand)
  endif()
  list(APPEND static_depends PACKAGE rocblas)
  list(APPEND static_depends PACKAGE rocsparse)
  list(APPEND static_depends PACKAGE rocprim)
  list(APPEND static_depends PACKAGE rocrand)

  target_link_libraries(rocalution PRIVATE rocalution_hip hip::host)
endif()

set_target_properties(rocalution PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

if(WIN32 AND BUILD_CLIENTS AND BUILD_SHARED_LIBS)
  add_custom_command(TARGET rocalution POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocalution> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocalution>)
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command(TARGET rocalution POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/rocalution.pdb ${PROJECT_BINARY_DIR}/clients/staging/rocalution.pdb )
  endif()
endif()

# Install targets
if(SUPPORT_HIP)
  rocm_install_targets(TARGETS rocalution rocalution_hip
                       INCLUDE ${CMAKE_BINARY_DIR}/include
                       )
else()
  rocm_install_targets(TARGETS rocalution
                       INCLUDE ${CMAKE_BINARY_DIR}/include
                       )
endif()

# Export targets
rocm_export_targets(TARGETS roc::rocalution
                    DEPENDS
                      ${package_depends}
                    STATIC_DEPENDS
                      ${static_depends}
                    NAMESPACE roc::)
