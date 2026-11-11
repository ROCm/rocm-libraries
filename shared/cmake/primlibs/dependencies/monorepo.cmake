# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_guard(DIRECTORY)

include(${CMAKE_CURRENT_LIST_DIR}/rocm-cmake.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

# Adds the 'roc::' target names that hipCUB and rocThrust link against. This
# is called whether rocPRIM was fetched or was already part of the build, so
# this must be idempotent.
#
# This family of macros is called by fetch_monorepo_dep.
macro(provide_rocprim_targets)
  if(TARGET rocprim AND NOT TARGET roc::rocprim)
    add_library(roc::rocprim ALIAS rocprim)
  endif()
  if(TARGET rocprim_hip AND NOT TARGET roc::rocprim_hip)
    add_library(roc::rocprim_hip ALIAS rocprim_hip)
  endif()
endmacro()

# Adds the 'roc::rocrand' target name that rocPRIM tests and rocThrust
# benchmarks link against, with the same idempotency requirement as
# provide_rocprim_targets().
macro(provide_rocrand_targets)
  if(TARGET rocrand AND NOT TARGET roc::rocrand)
    add_library(roc::rocrand ALIAS rocrand)
  endif()
endmacro()

# Fetches and configures rocPRIM for a given source directory.
#
# This family of macros is called by fetch_monorepo_dep.
macro(configure_rocprim _source_dir)
  # rocPRIM is a header-only library. 'BUILD_SHARED_LIBS' has no effect on it.
  FetchContent_Declare(
    prim
    SOURCE_DIR    ${_source_dir}
    INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(prim)
endmacro()

# Fetches and configures rocRAND for a given source directory.
macro(configure_rocrand _source_dir)
  # rocRAND should share 'BUILD_SHARED_LIBS' similarly to the consuming library.
  FetchContent_Declare(
    rocrand
    SOURCE_DIR    ${_source_dir}
    INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand
    EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(rocrand)
endmacro()

# Fetches <package> unless this build already provides it.
#
# Fetching is skipped if the depedency is already part of the build. This is
# specifically for hipCCL and TheRock-based builds. The guard is here since
# the target should be reused regardless of fetch method.
# * MONOREPO would include the previously added source tree again.
# * PACKAGE can silently bind the consumer to an installed copy instead of
#   the intended target provided by TheRock.
macro(fetch_monorepo_dep)
  set(args PACKAGE VERSION)
  cmake_parse_arguments(arg_fetch "" "${args}" "" ${ARGN})

  # Already part of the build under one of the usual target spellings?
  set(_dep_target_names
    ${arg_fetch_PACKAGE}
    roc::${arg_fetch_PACKAGE}
    ${arg_fetch_PACKAGE}::${arg_fetch_PACKAGE}
  )
  set(_dep_target_found FALSE)
  foreach(_dep_target_name ${_dep_target_names})
    if(TARGET ${_dep_target_name})
      set(_dep_target_found TRUE)
    endif()
  endforeach()

  # Skip if target already available, otherwise fetch it.
  if(_dep_target_found)
    message(STATUS "The '${arg_fetch_PACKAGE}' target already available.")
  else()
    fetch_monorepo_dep_impl(${ARGN})
  endif()

  # Whatever the outcome, the consumers need the same target names.
  if(COMMAND "provide_${arg_fetch_PACKAGE}_targets")
    cmake_language(CALL "provide_${arg_fetch_PACKAGE}_targets")
  endif()
endmacro()

# The actual fetching for fetch_monorepo_dep().
#
# TODO: once we hit CMake >= 3.24 change macro to function. This is because
# CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL to promote imported
# targets of find_package to the global scope.
macro(fetch_monorepo_dep_impl)
  set(args PACKAGE VERSION)
  cmake_parse_arguments(arg_fetch "" "${args}" "" ${ARGN})

  string(TOUPPER "${arg_fetch_PACKAGE}_FETCH_METHOD" fetch_method_name)
  if(NOT DEFINED ${fetch_method_name})
    set(
      ${fetch_method_name} "PACKAGE"
      CACHE STRING
      "The method used to fetch ${arg_fetch_PACKAGE}"
    )
  endif()
  message(STATUS "Fetching ${arg_fetch_PACKAGE} via ${${fetch_method_name}}")
  if(${fetch_method_name} STREQUAL "PACKAGE")
    # VERSION is optional and if unset will expand to nothing
    find_package(${arg_fetch_PACKAGE} ${arg_fetch_VERSION} CONFIG QUIET)
    if(${arg_fetch_PACKAGE}_FOUND)
      message(STATUS "Found ${arg_fetch_PACKAGE} ${arg_fetch_VERSION}")
    else()
      message(FATAL_ERROR
        "Could not find ${arg_fetch_PACKAGE}. Consider installing it or "
        "setting ${fetch_method_name} to 'MONOREPO'"
      )
    endif()
  elseif(${fetch_method_name} STREQUAL "MONOREPO")
    execute_process(
      COMMAND
      "git" "rev-parse" "--show-toplevel"
      COMMAND_ERROR_IS_FATAL ANY
      OUTPUT_VARIABLE monorepo_root
    )
    string(REGEX MATCH "^[^\r\n]*" monorepo_root "${monorepo_root}")
    cmake_path(
      RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR
      BASE_DIRECTORY "${monorepo_root}/"
      OUTPUT_VARIABLE current_relative_path
    )
    # Check if the current source directory is a hipccl directory.
    set(is_currently_in_hipccl FALSE)
    if ("${current_relative_path}" MATCHES "hipccl")
      set(is_currently_in_hipccl TRUE)
    endif()

    # Check if the required sub project is in hipccl.
    set(is_hipccl_subproject FALSE)
    if(
      ${arg_fetch_PACKAGE} STREQUAL "rocprim" OR
      ${arg_fetch_PACKAGE} STREQUAL "hipcub" OR
      ${arg_fetch_PACKAGE} STREQUAL "rocthrust" OR
      ${arg_fetch_PACKAGE} STREQUAL "libhipcxx"
    )
      set(is_hipccl_subproject TRUE)
    endif()

    # Get the absolute path of the required subproject.
    if (${is_hipccl_subproject} AND ${is_currently_in_hipccl})
      # for hipCCL we can get the right hipccl{2,3} directory by
      # Using the relative direcotry. This does not work when we
      # are not in a hipccl sub directory.
      get_filename_component(
        dep_source_dir
        "${CMAKE_CURRENT_SOURCE_DIR}/../${repo_root_dir}/${arg_fetch_PACKAGE}"
        ABSOLUTE
      )
    elseif(NOT ${is_hipccl_subproject})
      # For non-hipCCL depedencies, we can get it from the repo
      # root directory.
      get_filename_component(
        dep_source_dir
        "${monorepo_root}/projects/${arg_fetch_PACKAGE}"
        ABSOLUTE
      )
    endif()

    message(STATUS "Looking for '${arg_fetch_PACKAGE}' in: ${dep_source_dir}")
    if(NOT IS_DIRECTORY "${dep_source_dir}")
      # The source directory of the dependency does not exist in the monorepo.
      # Maybe this is a sparse checkout and we need to add it.
      #
      # This requires git!
      find_git()

      # Add the project directory to the sparse checkout.
      execute_process(
        COMMAND
        ${GIT_PATH} "sparse-checkout" "add" "${dep_source_dir}"
        COMMAND_ERROR_IS_FATAL ANY
      )
    endif()

    # TODO: remove this once all monorepo dependencies stop throwing warnings
    override_variable(ROCM_DISABLE_CHECKS ON)
    # Don't generate developer targets
    override_variable(BUILD_BENCHMARK OFF)
    override_variable(BUILD_EXAMPLE OFF)
    override_variable(BUILD_TEST OFF)

    cmake_language(
      CALL "configure_${arg_fetch_PACKAGE}" "${dep_source_dir}"
    )

    restore_variable(BUILD_TEST)
    restore_variable(BUILD_EXAMPLE)
    restore_variable(BUILD_BENCHMARK)
    restore_variable(ROCM_DISABLE_CHECKS)
  else()
    message(FATAL_ERROR
      "Unknown value ${${fetch_method_name}} in ${fetch_method_name}. Valid "
      "values are: PACKAGE, MONOREPO."
    )
  endif()
endmacro()
