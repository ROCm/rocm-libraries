# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fortran bindings for hipSPARSE. hipsparse.F90 is generated from the C headers:
# regenerate it, do not edit it.

cmake_minimum_required(VERSION 3.19)

project(hipsparse-fortran LANGUAGES NONE)

# Not ON: ROCm ships no Fortran compiler on Windows.
option(BUILD_FORTRAN_BINDINGS "Build the hipSPARSE Fortran bindings" "${UNIX}")
option(BUILD_FORTRAN_CLIENTS "Build the hipSPARSE Fortran binding tests" ON)

set(FORTRAN_ARRAY_INTERFACES "assumed-shape" CACHE STRING
    "Array-argument overloads to expose: none, assumed-shape, or assumed-rank")
set_property(CACHE FORTRAN_ARRAY_INTERFACES PROPERTY STRINGS
             none assumed-shape assumed-rank)

if(NOT BUILD_FORTRAN_BINDINGS)
  return()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmFortranCompiler.cmake")
if(NOT CMAKE_Fortran_COMPILER)
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

# Each module compiles to one object, and a linker extracts whole objects from
# an archive, so a program that calls one wrapper links all of them. One section
# per procedure lets a program linked with -Wl,--gc-sections keep only the
# wrappers it calls, and still link when another wrapper names a C function the
# installed library does not export. Only these compilers split the sections
# (the Intel ones accept the flags but do not).
target_compile_options(hipsparse_fortran PRIVATE
  "$<$<COMPILE_LANG_AND_ID:Fortran,GNU,LLVMFlang,Flang,NVHPC>:-ffunction-sections;-fdata-sections>")

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
enable_language(C)
find_package(hipsparse QUIET)

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

# The uninstrumented Fortran executables cannot link an ASan libhipsparse.
if(BUILD_FORTRAN_CLIENTS AND BUILD_ADDRESS_SANITIZER)
  message(STATUS
    "hipSPARSE: address sanitizer is enabled, skipping the Fortran binding tests.")
  set(BUILD_FORTRAN_CLIENTS OFF)
endif()

if(BUILD_FORTRAN_CLIENTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)

set(_fortran_pkgdir "${CMAKE_INSTALL_LIBDIR}/cmake/hipsparse-fortran")

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

# Read from this tree, not hipsparse_VERSION, which is the INSTALLED hipSPARSE's.
set(_fortran_version "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipsparse/CMakeLists.txt")
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../../projects/hipsparse/CMakeLists.txt" _hipsparse_root)
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
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hipsparse-fortran-config-version.cmake"
          DESTINATION "${_fortran_pkgdir}"
          COMPONENT devel)
else()
  message(STATUS
    "hipSPARSE: could not determine the version for the Fortran package, so no "
    "hipsparse-fortran-config-version.cmake is installed; "
    "find_package(hipsparse-fortran <version>) will not match.")
endif()
