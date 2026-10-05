# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fortran bindings for hipSPARSE. hipsparse.F90 is generated from the C headers:
# regenerate it, do not edit it.

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(hipsparse-fortran LANGUAGES NONE)
endif()

# Not on Windows (ROCm ships no Fortran compiler there). Not on CUDA either:
# hipsparse.F90 is generated from the ROCm path, and the module object
# references entry points a cuSPARSE-backed libhipsparse does not define.
if(HIPSPARSE_ENABLE_CUDA)
  set(_hipsparse_fortran_default OFF)
else()
  set(_hipsparse_fortran_default "${UNIX}")
endif()
option(BUILD_FORTRAN_BINDINGS "Build the hipSPARSE Fortran bindings"
       "${_hipsparse_fortran_default}")
unset(_hipsparse_fortran_default)

# Gates only the binding tests; the Fortran samples follow HIPSPARSE_ENABLE_FORTRAN.
option(BUILD_FORTRAN_CLIENTS "Build the hipSPARSE Fortran binding tests" ON)

# The BUILD_FORTRAN_* cache entries are shared by the whole monorepo; these
# single hipSPARSE out.
if(DEFINED HIPSPARSE_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${HIPSPARSE_BUILD_FORTRAN_BINDINGS})
endif()
if(DEFINED HIPSPARSE_BUILD_FORTRAN_CLIENTS)
  set(BUILD_FORTRAN_CLIENTS ${HIPSPARSE_BUILD_FORTRAN_CLIENTS})
endif()

# The two upper tiers are alternatives: assumed-rank replaces the per-rank
# overloads. Routines without array overloads keep type(c_ptr) under all three.
set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# PARENT_SCOPE warns at the top level, so only in-tree.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIPSPARSE_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

if(HIPSPARSE_ENABLE_CUDA AND NOT FORTRAN_ARRAY_INTERFACES STREQUAL "none")
  message(WARNING
    "hipSPARSE: the Fortran bindings were requested on the CUDA backend. "
    "hipsparse.F90 is generated from the ROCm path and the "
    "'${FORTRAN_ARRAY_INTERFACES}' tier references 361 entry points, around "
    "119 of which a CUDA 12.x libhipsparse does not define (44 on CUDA 11.8, "
    "242 on CUDA 13); anything that uses the module will fail to link. "
    "Configure with -DFORTRAN_ARRAY_INTERFACES=none, which emits no such "
    "references, or leave BUILD_FORTRAN_BINDINGS at its default.")
endif()

# ROCM_LIBS_HAVE_FORTRAN is the root's current name; drop it once the root
# publishes ROCM_HAVE_FORTRAN.
if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
elseif(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
else()
  include(CheckLanguage)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")

  # rocm-cmake's ROCMChecks watches CMAKE_Fortran_COMPILER and fails under
  # ROCM_ERROR_TOOLCHAIN_VAR=ON; silence it across the probe.
  set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
  set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
  set(ROCM_WARN_TOOLCHAIN_VAR OFF)
  set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

  # Before check_language(), which is a no-op once a compiler is set.
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
  message(STATUS "hipSPARSE: no Fortran compiler found, skipping the Fortran bindings")
  return()
endif()

enable_language(Fortran)

include(GNUInstallDirs)

if(NOT FORTRAN_ARRAY_INTERFACES MATCHES "^(none|assumed-shape|assumed-rank)$")
  message(FATAL_ERROR
    "FORTRAN_ARRAY_INTERFACES must be none, assumed-shape or assumed-rank, "
    "not '${FORTRAN_ARRAY_INTERFACES}'.")
endif()

add_library(hipsparse_fortran STATIC hipsparse.F90)

add_library(roc::hipsparse_fortran ALIAS hipsparse_fortran)

# A .mod is compiler-specific, so artifacts install per compiler, named as the
# user types it. Must match the mapping in hipsparse-fortran-config.cmake.in.
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
set(HIPSPARSE_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(hipsparse_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(hipsparse_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(hipsparse_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(hipsparse_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPSPARSE_FORTRAN_COMPILER_DIR}>")

# Link interface only. hipsparse-config.cmake's find_dependency(hip) needs C or
# CXX enabled (FindThreads).
if(NOT TARGET roc::hipsparse)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(hipsparse QUIET)
endif()

# Substituted into an if() in the installed config, so never left empty.
if(TARGET roc::hipsparse)
  target_link_libraries(hipsparse_fortran INTERFACE roc::hipsparse)
  set(HIPSPARSE_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  set(HIPSPARSE_FORTRAN_NEEDS_C_PACKAGE FALSE)
  message(STATUS
    "hipSPARSE: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lhipsparse) itself.")
endif()

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HIPSPARSE_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
endif()

# In-tree the binding tests follow the root's client/test switches; a plain
# variable, so the shared cache entry is not written back.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND BUILD_FORTRAN_CLIENTS
   AND NOT (HIPSPARSE_ENABLE_CLIENT OR HIPSPARSE_BUILD_TESTING OR BUILD_TESTING))
  message(STATUS
    "hipSPARSE: clients are not being built, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

# The uninstrumented Fortran executables cannot link an ASan libhipsparse.
if(BUILD_FORTRAN_CLIENTS AND HIPSPARSE_ENABLE_ASAN)
  message(STATUS
    "hipSPARSE: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

# A clients-only build links the installed library: keep the binding out of
# its devel package.
if(DEFINED HIPSPARSE_ENABLE_HOST AND NOT HIPSPARSE_ENABLE_HOST)
  return()
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hipsparse")

# Plain install(): standalone, rocm-cmake is not loaded.
install(TARGETS hipsparse_fortran
        EXPORT hipsparse-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${HIPSPARSE_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${HIPSPARSE_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/, not include/fortran/, which the config lists as compilers.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/fortran"
        COMPONENT devel)
install(EXPORT hipsparse-fortran-targets
        FILE hipsparse-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(HIPSPARSE_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(HIPSPARSE_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  PATH_VARS HIPSPARSE_FORTRAN_LIBDIR HIPSPARSE_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# find_package(hipsparse-fortran) searches lib/cmake/hipsparse-fortran*.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/hipsparse-fortran-config.cmake"
"# Generated forwarder. The real config lives beside the hipSPARSE C package, in
# lib/cmake/hipsparse, so that hipsparse-config.cmake can include it as a sibling.
# This file only exists so that find_package(hipsparse-fortran) resolves to the
# same place.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../hipsparse/hipsparse-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/hipsparse-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipsparse-fortran"
        COMPONENT devel)

# Read from this tree's root, not hipsparse_VERSION, which standalone reports
# the INSTALLED hipSPARSE.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _hipsparse_root)
  string(REGEX MATCH "project[ \t]*\\([ \t]*hipsparse[ \t]+VERSION[ \t]+([0-9]+\\.[0-9]+\\.[0-9]+)"
         _m "${_hipsparse_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_hipsparse_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  # Beside both configs: CMake reads the version file next to the one it found.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/hipsparse-fortran"
          COMPONENT devel)
else()
  message(STATUS
    "hipSPARSE: could not determine the version for the Fortran package, so no "
    "hipsparse-fortran-config-version.cmake is installed; "
    "find_package(hipsparse-fortran <version>) will not match.")
endif()
