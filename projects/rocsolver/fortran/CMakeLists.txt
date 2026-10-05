# ##########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

# rocsolver.F90 is generated from the rocSOLVER headers; regenerate it, do not edit it.
# It says `use rocblas`, so a rocblas.mod from the same compiler must exist to compile it.

cmake_minimum_required(VERSION 3.13)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocsolver-fortran LANGUAGES NONE)
endif()

# ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the rocSOLVER Fortran bindings" "${UNIX}")

option(BUILD_FORTRAN_CLIENTS "Build the rocSOLVER Fortran clients and binding tests" ON)

# Both cache entries are shared monorepo-wide; the per-library names override them here only.
if(DEFINED ROCSOLVER_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${ROCSOLVER_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED ROCSOLVER_BUILD_FORTRAN_CLIENTS)
  set(BUILD_FORTRAN_CLIENTS ${ROCSOLVER_BUILD_FORTRAN_CLIENTS})
endif()

# Accepted for uniformity with the other bindings; rocsolver.F90 has no array overloads.
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Guarded: PARENT_SCOPE at the top level warns under -Wdev.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCSOLVER_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
elseif(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
else()
  include(CheckLanguage)

  # check_language() writes the CMAKE_Fortran_COMPILER cache entry, which
  # rocm-cmake's toolchain-variable watch reports (and fails on under CI).
  set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
  set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
  set(ROCM_WARN_TOOLCHAIN_VAR OFF)
  set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

  # Must run before check_language(), which keeps any compiler already set.
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/ROCmFortranCompiler.cmake)
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
if(NOT TARGET roc::rocblas_fortran)
  find_package(rocblas-fortran QUIET CONFIG PATH_SUFFIXES lib/cmake/rocblas)
endif()
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
  # Ignored before CMake 3.18; the .F90 suffix is preprocessed anyway.
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
if(NOT TARGET roc::rocsolver)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocsolver QUIET)
endif()

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

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCSOLVER_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# In-tree the tests follow rocSOLVER's client switches, which also enable testing at the root.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND BUILD_FORTRAN_CLIENTS
   AND NOT (BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES))
  message(STATUS
    "rocSOLVER: clients are not being built, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
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

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver")

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

# Lets find_package(rocsolver-fortran) reach the config installed in lib/cmake/rocsolver.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/rocsolver-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the rocSOLVER C package, in
# lib/cmake/rocsolver, so one CMAKE_PREFIX_PATH entry finds both.
# This file only exists so that find_package(rocsolver-fortran) resolves to the
# same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../rocsolver/rocsolver-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/rocsolver-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver-fortran"
        COMPONENT devel)

# Read from this tree, not rocsolver_VERSION, which standalone is the INSTALLED rocSOLVER's.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _rocsolver_root)
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
  # Both directories: the version file is looked up beside whichever config resolved.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocsolver-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocsolver-fortran"
          COMPONENT devel)
else()
  message(STATUS
    "rocSOLVER: could not determine the version for the Fortran package, so no "
    "rocsolver-fortran-config-version.cmake is installed; "
    "find_package(rocsolver-fortran <version>) will not match.")
endif()
