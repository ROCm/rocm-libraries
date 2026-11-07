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

# rocsolver.F90 is generated from the rocSOLVER headers; regenerate it, do not edit it.
# It says `use rocblas`, so a rocblas.mod from the same compiler must exist to compile it.

cmake_minimum_required(VERSION 3.19)

project(rocsolver-fortran LANGUAGES NONE)

# Not ON: ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the rocSOLVER Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the rocSOLVER Fortran binding tests" ON)

set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
  message(STATUS "rocSOLVER: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

include(GNUInstallDirs)

# A missing optional binding must skip, not take the C library down with it.
find_package(rocblas-fortran QUIET CONFIG)
if(NOT TARGET roc::rocblas_fortran)
  message(STATUS
    "rocSOLVER: the rocBLAS Fortran binding was not found, skipping the "
    "rocSOLVER Fortran bindings. They are written against it (`use rocblas`), "
    "so rocblas.mod must be built with ${CMAKE_Fortran_COMPILER} and reachable "
    "through CMAKE_PREFIX_PATH.")
  return()
endif()

add_library(rocsolver_fortran STATIC rocsolver.F90)

add_library(roc::rocsolver_fortran ALIAS rocsolver_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(rocsolver_fortran PRIVATE
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>")

# A .mod is compiler-specific, so artifacts install per compiler, named as the user types it.
# Keep in step with the same mapping in rocsolver-fortran-config.cmake.in.
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

# Versions of one compiler share a directory; override (e.g. gfortran-13.3.0) to keep several.
set(ROCSOLVER_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(rocsolver_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(rocsolver_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(rocsolver_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(rocsolver_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}>")

# PUBLIC: a consumer's `use rocsolver` needs rocblas.mod on its include path too.
target_link_libraries(rocsolver_fortran PUBLIC roc::rocblas_fortran)

# Link interface only. C is enabled because hip-config.cmake reaches FindThreads.
enable_language(C)
find_package(rocsolver QUIET)

# Substituted into a bare if() in the config template, so it must be TRUE or FALSE.
if(TARGET roc::rocsolver)
  target_link_libraries(rocsolver_fortran INTERFACE roc::rocsolver)
  set(ROCSOLVER_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  set(ROCSOLVER_FORTRAN_NEEDS_C_PACKAGE FALSE)
  message(STATUS
    "rocSOLVER: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lrocsolver) itself.")
endif()

# The uninstrumented Fortran executables cannot link an ASan librocsolver.
if(BUILD_FORTRAN_CLIENTS AND BUILD_ADDRESS_SANITIZER)
  message(STATUS
    "rocSOLVER: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver-fortran")

# Plain install(), not rocm_install(): this directory also configures without rocm-cmake.
install(TARGETS rocsolver_fortran
        EXPORT rocsolver-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCSOLVER_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/: every entry under include/fortran/ is read as a compiler name.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsolver/fortran"
        COMPONENT devel)
install(EXPORT rocsolver-fortran-targets
        FILE rocsolver-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(ROCSOLVER_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCSOLVER_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocsolver-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # Not the compiler directory: it is a bare name, which PATH_VARS would resolve under the prefix.
  PATH_VARS ROCSOLVER_FORTRAN_LIBDIR ROCSOLVER_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# Read from this tree, not rocsolver_VERSION, which standalone is the INSTALLED rocSOLVER's.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/rocsolver/CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/rocsolver/CMakeLists.txt" _rocsolver_root)
  string(REGEX MATCH "set\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_rocsolver_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_rocsolver_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
else()
  message(STATUS
    "rocSOLVER: could not determine the version for the Fortran package, so no "
    "rocsolver-fortran-config-version.cmake is installed; "
    "find_package(rocsolver-fortran <version>) will not match.")
endif()
