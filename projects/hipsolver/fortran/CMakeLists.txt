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

# hipsolver.F90 is generated from the hipSOLVER headers; regenerate it, do not edit it.

cmake_minimum_required(VERSION 3.14)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hipsolver-fortran LANGUAGES NONE)
endif()

# Off on the CUDA backend: hipsolver.F90 is not tested against cuSOLVER. The test
# repeats the root's pre-project() one for a standalone configure; keep them identical.
if(NOT DEFINED _hipsolver_cuda_backend)
  set(_hipsolver_cuda_backend OFF)
  if(DEFINED USE_CUDA)
    if(USE_CUDA)
      set(_hipsolver_cuda_backend ON)
    endif()
  elseif("$ENV{HIP_PLATFORM}" STREQUAL "nvidia" OR "${HIP_PLATFORM}" STREQUAL "nvidia")
    set(_hipsolver_cuda_backend ON)
  endif()
endif()

if(_hipsolver_cuda_backend)
  set(_hipsolver_fortran_default OFF)
else()
  # ROCm ships no Fortran compiler on Windows.
  set(_hipsolver_fortran_default "${UNIX}")
endif()
option(BUILD_FORTRAN_BINDINGS "Build the hipSOLVER Fortran bindings"
       "${_hipsolver_fortran_default}")
unset(_hipsolver_fortran_default)

option(BUILD_FORTRAN_CLIENTS "Build the hipSOLVER Fortran clients and binding tests" ON)

# Both cache entries are shared monorepo-wide; the per-library names override them here only.
if(DEFINED HIPSOLVER_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${HIPSOLVER_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIPSOLVER_BUILD_FORTRAN_CLIENTS)
  set(BUILD_FORTRAN_CLIENTS ${HIPSOLVER_BUILD_FORTRAN_CLIENTS})
endif()

# Accepted for uniformity with the other bindings; hipsolver.F90 has no array overloads.
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Read by clients/. Guarded: PARENT_SCOPE at the top level warns under -Wdev.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIPSOLVER_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

if(_hipsolver_cuda_backend)
  message(WARNING
    "hipSOLVER: the Fortran bindings were requested on the CUDA backend. "
    "hipsolver.F90 is generated from the ROCm code path and is not tested "
    "against a cuSOLVER-backed libhipsolver, which is why it is off by default "
    "there. Building it is supported; vouching for it is not.")
endif()

if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
elseif(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
else()
  include(CheckLanguage)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")

  # check_language() writes the CMAKE_Fortran_COMPILER cache entry, which
  # rocm-cmake's toolchain-variable watch reports (and fails on under CI).
  set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
  set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
  set(ROCM_WARN_TOOLCHAIN_VAR OFF)
  set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

  # Must run before check_language(), which keeps any compiler already set.
  rocm_fortran_prefer_amdflang()

  check_language(Fortran)
  set(ROCM_WARN_TOOLCHAIN_VAR "${_rocm_warn_toolchain_var}")
  set(ROCM_ERROR_TOOLCHAIN_VAR "${_rocm_error_toolchain_var}")
  unset(_rocm_warn_toolchain_var)
  unset(_rocm_error_toolchain_var)

  if(CMAKE_Fortran_COMPILER)
    set(_have_fortran TRUE)
  else()
    set(_have_fortran FALSE)
  endif()
endif()

if(NOT _have_fortran)
  message(STATUS "hipSOLVER: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

include(GNUInstallDirs)

add_library(hipsolver_fortran STATIC hipsolver.F90)

add_library(roc::hipsolver_fortran ALIAS hipsolver_fortran)

# A .mod is compiler-specific, so artifacts install per compiler, named as the user types it.
# Keep in step with the same mapping in hipsolver-fortran-config.cmake.in.
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
set(HIPSOLVER_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(hipsolver_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  # Ignored before CMake 3.18; the .F90 suffix is preprocessed anyway.
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(hipsolver_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hipsolver_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(hipsolver_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPSOLVER_FORTRAN_COMPILER_DIR}>")

# Link interface only. C is enabled because hip-config.cmake reaches FindThreads.
if(NOT TARGET roc::hipsolver)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipsolver QUIET)
endif()

# Substituted into a bare if() in the config template, so it must be TRUE or FALSE.
if(TARGET roc::hipsolver)
  target_link_libraries(hipsolver_fortran INTERFACE roc::hipsolver)
  set(HIPSOLVER_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  set(HIPSOLVER_FORTRAN_NEEDS_C_PACKAGE FALSE)
  message(STATUS
    "hipSOLVER: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lhipsolver) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIPSOLVER_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# In-tree the tests follow hipSOLVER's client switches, which also enable testing at the root.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND BUILD_FORTRAN_CLIENTS
   AND NOT (BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS))
  message(STATUS
    "hipSOLVER: clients are not being built, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# The uninstrumented Fortran executables cannot link an ASan libhipsolver.
if(BUILD_FORTRAN_CLIENTS AND BUILD_ADDRESS_SANITIZER)
  message(STATUS
    "hipSOLVER: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver")

# Plain install(), not rocm_install(): this directory also configures without rocm-cmake.
install(TARGETS hipsolver_fortran
        EXPORT hipsolver-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIPSOLVER_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPSOLVER_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/: every entry under include/fortran/ is read as a compiler name.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipsolver.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsolver/fortran"
        COMPONENT devel)
install(EXPORT hipsolver-fortran-targets
        FILE hipsolver-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(HIPSOLVER_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIPSOLVER_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hipsolver-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsolver-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # Not the compiler directory: it is a bare name, which PATH_VARS would resolve under the prefix.
  PATH_VARS HIPSOLVER_FORTRAN_LIBDIR HIPSOLVER_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsolver-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# Lets find_package(hipsolver-fortran) reach the config installed in lib/cmake/hipsolver.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/hipsolver-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the hipSOLVER C package, in
# lib/cmake/hipsolver, so one CMAKE_PREFIX_PATH entry finds both.
# This file only exists so that find_package(hipsolver-fortran) resolves to the
# same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../hipsolver/hipsolver-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/hipsolver-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver-fortran"
        COMPONENT devel)

# Read from this tree, not hipsolver_VERSION, which standalone is the INSTALLED hipSOLVER's.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _hipsolver_root)
  string(REGEX MATCH "set\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_hipsolver_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_hipsolver_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipsolver-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  # Both directories: the version file is looked up beside whichever config resolved.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsolver-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsolver-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipsolver-fortran"
          COMPONENT devel)
else()
  message(STATUS
    "hipSOLVER: could not determine the version for the Fortran package, so no "
    "hipsolver-fortran-config-version.cmake is installed; "
    "find_package(hipsolver-fortran <version>) will not match.")
endif()
