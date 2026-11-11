# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# SPDX-License-Identifier: MIT
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


# Fortran bindings for rocSPARSE. rocsparse.F90 is generated from the C headers:
# regenerate it, do not edit it.

cmake_minimum_required(VERSION 3.19)

project(rocsparse-fortran LANGUAGES NONE)

# Not ON: ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the rocSPARSE Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the rocSPARSE Fortran binding tests" ON)

set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# An installed rocSPARSE does not record that it was built with
# BUILD_ROCSPARSE_ILP64, so the bindings cannot detect it.
option(ROCSPARSE_ILP64
       "The installed rocSPARSE was built with BUILD_ROCSPARSE_ILP64" OFF)

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

if(ROCSPARSE_ILP64)
  message(STATUS
    "rocSPARSE: ROCSPARSE_ILP64 widens rocsparse_int to int64_t, which the "
    "generated module binds as integer(c_int); skipping the Fortran bindings "
    "rather than building one that is silently ABI-incompatible.")
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
  message(STATUS "rocSPARSE: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(rocsparse_fortran STATIC rocsparse.F90)

add_library(roc::rocsparse_fortran ALIAS rocsparse_fortran)

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(rocsparse_fortran PRIVATE
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>")

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Must match the mapping in rocsparse-fortran-config.cmake.in.
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

# Versions of one compiler share a directory; set this (e.g. gfortran-13) to
# install several side by side.
set(ROCSPARSE_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(rocsparse_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(rocsparse_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(rocsparse_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(rocsparse_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCSPARSE_FORTRAN_COMPILER_DIR}>")

# Link interface only. rocsparse-config.cmake's find_dependency(hip) needs C or
# CXX enabled (FindThreads).
enable_language(C)
find_package(rocsparse QUIET)

# Substituted into an if() in the installed config, so never left empty.
if(TARGET roc::rocsparse)
  target_link_libraries(rocsparse_fortran INTERFACE roc::rocsparse)
  set(ROCSPARSE_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  set(ROCSPARSE_FORTRAN_NEEDS_C_PACKAGE FALSE)
  message(STATUS
    "rocSPARSE: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lrocsparse) itself.")
endif()

# The uninstrumented Fortran executables cannot link an ASan librocsparse.
if(BUILD_FORTRAN_CLIENTS AND BUILD_ADDRESS_SANITIZER)
  message(STATUS
    "rocSPARSE: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/rocsparse-fortran")

install(TARGETS rocsparse_fortran
        EXPORT rocsparse-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCSPARSE_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCSPARSE_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/, not include/fortran/, which the config lists as compilers.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocsparse.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocsparse/fortran"
        COMPONENT devel)
install(EXPORT rocsparse-fortran-targets
        FILE rocsparse-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(ROCSPARSE_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCSPARSE_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocsparse-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocsparse-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  PATH_VARS ROCSPARSE_FORTRAN_LIBDIR ROCSPARSE_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsparse-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# Read from this tree, not rocsparse_VERSION, which is the INSTALLED rocSPARSE's.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/rocsparse/CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/rocsparse/CMakeLists.txt" _rocsparse_root)
  string(REGEX MATCH "set\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_rocsparse_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_rocsparse_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocsparse-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsparse-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
else()
  message(STATUS
    "rocSPARSE: could not determine the version for the Fortran package, so no "
    "rocsparse-fortran-config-version.cmake is installed; "
    "find_package(rocsparse-fortran <version>) will not match.")
endif()
