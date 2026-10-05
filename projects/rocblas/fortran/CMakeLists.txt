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

# Fortran bindings for rocBLAS. rocblas.F90 is GENERATED from the C headers by
# the rocm-fortran generator: do not edit it, regenerate it.

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rocblas-fortran LANGUAGES NONE)
endif()

# Not ON: ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the rocBLAS Fortran bindings" "${UNIX}")

# BUILD_FORTRAN_BINDINGS is shared by the whole monorepo; this one wins here.
if(DEFINED ROCBLAS_BUILD_FORTRAN_BINDINGS)
  set(BUILD_FORTRAN_BINDINGS ${ROCBLAS_BUILD_FORTRAN_BINDINGS})
endif()

set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

# Standalone, PARENT_SCOPE at top level fails under -Werror=dev.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCBLAS_HAVE_FORTRAN_BINDINGS FALSE PARENT_SCOPE)
endif()

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

# ROCM_LIBS_HAVE_FORTRAN is the root's current name; drop it once the root
# publishes ROCM_HAVE_FORTRAN.
if(DEFINED ROCM_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_HAVE_FORTRAN})
elseif(DEFINED ROCM_LIBS_HAVE_FORTRAN)
  set(_have_fortran ${ROCM_LIBS_HAVE_FORTRAN})
else()
  include(CheckLanguage)

  # rocm-cmake's ROCMChecks would flag the CMAKE_Fortran_COMPILER cache write
  # below (a hard error under ROCM_ERROR_TOOLCHAIN_VAR); silence it for this scope.
  set(_rocm_warn_toolchain_var "${ROCM_WARN_TOOLCHAIN_VAR}")
  set(_rocm_error_toolchain_var "${ROCM_ERROR_TOOLCHAIN_VAR}")
  set(ROCM_WARN_TOOLCHAIN_VAR OFF)
  set(ROCM_ERROR_TOOLCHAIN_VAR OFF)

  # Prefer ROCm's amdflang over the first compiler on PATH (often gfortran); an
  # explicit CMAKE_Fortran_COMPILER or FC wins. Must precede check_language().
  if(NOT CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
    set(_rocm_fc_hints)
    foreach(_rocm_fc_cc "${CMAKE_CXX_COMPILER}" "${CMAKE_C_COMPILER}"
                        "${CMAKE_HIP_COMPILER}")
      if(_rocm_fc_cc)
        get_filename_component(_rocm_fc_dir "${_rocm_fc_cc}" DIRECTORY)
        list(APPEND _rocm_fc_hints "${_rocm_fc_dir}")
      endif()
    endforeach()
    foreach(_rocm_fc_root "${ROCM_PATH}" "$ENV{ROCM_PATH}" "/opt/rocm")
      if(_rocm_fc_root)
        list(APPEND _rocm_fc_hints "${_rocm_fc_root}/bin"
                                   "${_rocm_fc_root}/lib/llvm/bin")
      endif()
    endforeach()
    # Uncached, so a changed ROCM_PATH is honoured on reconfigure.
    unset(_rocm_amdflang CACHE)
    find_program(_rocm_amdflang NAMES amdflang HINTS ${_rocm_fc_hints}
                 NO_DEFAULT_PATH)
    # A ROCm tree can carry a dangling amdflang symlink.
    if(_rocm_amdflang)
      execute_process(COMMAND "${_rocm_amdflang}" --version
                      RESULT_VARIABLE _rocm_fc_rc
                      OUTPUT_QUIET ERROR_QUIET)
      if(_rocm_fc_rc EQUAL 0)
        set(CMAKE_Fortran_COMPILER "${_rocm_amdflang}" CACHE FILEPATH
            "Fortran compiler used for the ROCm Fortran bindings" FORCE)
      endif()
      unset(_rocm_fc_rc)
    endif()
    unset(_rocm_amdflang CACHE)
    unset(_rocm_fc_hints)
    unset(_rocm_fc_cc)
    unset(_rocm_fc_dir)
    unset(_rocm_fc_root)
  endif()

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
  message(STATUS "rocBLAS: no Fortran compiler found, skipping the Fortran bindings")
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

add_library(rocblas_fortran STATIC rocblas.F90)

add_library(roc::rocblas_fortran ALIAS rocblas_fortran)

# Keep in step with rocblas-fortran-config.cmake.in, which recomputes this name.
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
set(ROCBLAS_FORTRAN_COMPILER_DIR "${_fc_name}" CACHE STRING
    "Per-compiler subdirectory the .mod and the archive install under")

set(_moddir "${CMAKE_CURRENT_BINARY_DIR}/include")
set_target_properties(rocblas_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY "${_moddir}"
  Fortran_PREPROCESS ON
  Fortran_FORMAT FREE
  POSITION_INDEPENDENT_CODE ON)

if(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-shape")
  target_compile_definitions(rocblas_fortran PRIVATE USE_ASSUMED_SHAPE=1)
elseif(FORTRAN_ARRAY_INTERFACES STREQUAL "assumed-rank")
  target_compile_definitions(rocblas_fortran PRIVATE USE_ASSUMED_RANK=1)
endif()

target_include_directories(rocblas_fortran PUBLIC
  "$<BUILD_INTERFACE:${_moddir}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCBLAS_FORTRAN_COMPILER_DIR}>")

# Link interface only. rocblas-config's find_dependency(hip) reaches FindThreads,
# which needs C or CXX enabled; QUIET because the bindings build without rocBLAS.
if(NOT TARGET roc::rocblas)
  if(NOT CMAKE_C_COMPILER_LOADED AND NOT CMAKE_CXX_COMPILER_LOADED)
    enable_language(C)
  endif()
  find_package(rocblas QUIET)
endif()

# Never left empty: it is @-substituted into a bare if() in the installed config.
if(TARGET roc::rocblas)
  target_link_libraries(rocblas_fortran INTERFACE roc::rocblas)
  set(ROCBLAS_FORTRAN_NEEDS_C_PACKAGE TRUE)
else()
  message(STATUS
    "rocBLAS: the C library was not found, so the Fortran bindings carry no "
    "link interface; a consumer will have to link it (-lrocblas) itself.")
  set(ROCBLAS_FORTRAN_NEEDS_C_PACKAGE FALSE)
endif()

# The parent cannot see the compiler ID; it keys its libgfortran dependency on it.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCBLAS_HAVE_FORTRAN_BINDINGS TRUE PARENT_SCOPE)
  set(ROCBLAS_FORTRAN_COMPILER_ID "${CMAKE_Fortran_COMPILER_ID}" PARENT_SCOPE)
endif()

include(CMakePackageConfigHelpers)

# Beside rocblas-config.cmake, so one CMAKE_PREFIX_PATH entry finds both halves.
set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas")

install(TARGETS rocblas_fortran
        EXPORT rocblas-fortran-targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}/fortran/${ROCBLAS_FORTRAN_COMPILER_DIR}"
                COMPONENT devel)
install(DIRECTORY "${_moddir}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fortran/${ROCBLAS_FORTRAN_COMPILER_DIR}"
        COMPONENT devel
        FILES_MATCHING PATTERN "*.mod")

# Under share/, not include/fortran/, which the config enumerates as compilers.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rocblas.F90"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/rocblas/fortran"
        COMPONENT devel)
install(EXPORT rocblas-fortran-targets
        FILE rocblas-fortran-targets.cmake
        NAMESPACE roc::
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

set(ROCBLAS_FORTRAN_LIBDIR     "${CMAKE_INSTALL_LIBDIR}/fortran")
set(ROCBLAS_FORTRAN_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/fortran")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/rocblas-fortran-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/rocblas-fortran-config.cmake"
  INSTALL_DESTINATION "${_fortran_pkgdir}"
  # Not ROCBLAS_FORTRAN_COMPILER_DIR: it is a directory name, not a path.
  PATH_VARS ROCBLAS_FORTRAN_LIBDIR ROCBLAS_FORTRAN_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocblas-fortran-config.cmake"
        DESTINATION "${_fortran_pkgdir}"
        COMPONENT devel)

# find_package(rocblas-fortran) searches lib/cmake/rocblas-fortran*: forward it.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/forward/rocblas-fortran-config.cmake"
"# Generated forwarder to the config installed beside the rocBLAS C package.
include(\"\${CMAKE_CURRENT_LIST_DIR}/../rocblas/rocblas-fortran-config.cmake\")
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forward/rocblas-fortran-config.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas-fortran"
        COMPONENT devel)

# Not rocblas_VERSION: standalone, that is the INSTALLED rocBLAS's version.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _rocblas_root)
  string(REGEX MATCH "set[ \t]*\\([ \t]*VERSION_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
         _m "${_rocblas_root}")
  if(CMAKE_MATCH_1)
    set(_fortran_version "${CMAKE_MATCH_1}")
  endif()
  unset(_rocblas_root)
endif()

if(_fortran_version)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rocblas-fortran-config-version.cmake"
    VERSION "${_fortran_version}"
    COMPATIBILITY SameMajorVersion)
  # In both package dirs: CMake reads the version file beside the config it found.
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocblas-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rocblas-fortran-config-version.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/rocblas-fortran"
          COMPONENT devel)
else()
  message(STATUS
    "rocBLAS: could not determine the version for the Fortran package, so no "
    "rocblas-fortran-config-version.cmake is installed; "
    "find_package(rocblas-fortran <version>) will not match.")
endif()
