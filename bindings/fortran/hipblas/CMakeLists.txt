# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
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

# Fortran bindings for hipBLAS. hipblas.F90 is GENERATED from the C headers by
# the rocm-fortran generator: do not edit it, regenerate it.

cmake_minimum_required(VERSION 3.19)

project(hipblas-fortran LANGUAGES NONE)

# Not ON: ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the hipBLAS Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the hipBLAS Fortran binding tests" ON)

set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
  message(STATUS "hipBLAS: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

# Before the target: its INSTALL_INTERFACE include directory uses CMAKE_INSTALL_*.
include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(hipblas_fortran STATIC hipblas.F90)

add_library(roc::hipblas_fortran ALIAS hipblas_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(hipblas_fortran PRIVATE
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>")

# Keep in step with hipblas-fortran-config.cmake.in, which recomputes this name.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(_fc_name "gfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")     # Flang, LLVMFlang
  set(_fc_name "amdflang")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Cray")
  set(_fc_name "ftn")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  set(_fc_name "ifx")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(_fc_name "ifort")
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
  set(_fc_name "nvfortran")
else()
  get_filename_component(_fc_name "${CMAKE_Fortran_COMPILER}" NAME_WE)
  string(TOLOWER "${_fc_name}" _fc_name)
endif()

# A .mod is tied to the compiler version too: sites with several set this.
set(HIPBLAS_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(hipblas_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(hipblas_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hipblas_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(hipblas_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPBLAS_FORTRAN_COMPILER_DIR}>")

# Link interface only. hipblas-config's find_dependency(hip) reaches FindThreads,
# which needs C or CXX enabled; QUIET because the bindings build without hipBLAS.
enable_language(C)
find_package(hipblas QUIET)

# Never left empty: it is @-substituted into a bare if() in the installed config.
if(TARGET roc::hipblas)
  target_link_libraries(hipblas_fortran INTERFACE roc::hipblas)
  set(HIPBLAS_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  message(STATUS
    "hipBLAS: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lhipblas) itself.")
  set(HIPBLAS_FORTRAN_NEEDS_C_PACKAGE FALSE)
endif()

# The uninstrumented Fortran executables cannot link an ASan libhipblas.
if(BUILD_FORTRAN_CLIENTS AND BUILD_ADDRESS_SANITIZER)
  message(STATUS
    "hipBLAS: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hipblas-fortran")

install(TARGETS hipblas_fortran
        EXPORT hipblas-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIPBLAS_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPBLAS_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/, not include/fortran/, which the config enumerates as compilers.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipblas.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipblas/fortran"
        COMPONENT devel)
install(EXPORT hipblas-fortran-targets
        FILE hipblas-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(HIPBLAS_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIPBLAS_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hipblas-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hipblas-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # Not HIPBLAS_FORTRAN_COMPILER_DIR: it is a directory name, not a path.
  PATH_VARS HIPBLAS_FORTRAN_LIBDIR HIPBLAS_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipblas-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# Not hipblas_VERSION: standalone, that is the INSTALLED hipBLAS's version.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblas/CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipblas/CMakeLists.txt" _hipblas_root)
  string(REGEX MATCH "set[ \t]*\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_hipblas_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_hipblas_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipblas-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipblas-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
else()
  message(STATUS
    "hipBLAS: could not determine the version for the Fortran package, so no "
    "hipblas-fortran-config-version.cmake is installed; "
    "find_package(hipblas-fortran <version>) will not match.")
endif()
