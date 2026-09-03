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

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/rocm-cmake.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/utils.cmake)

# Fetches and configures rocPRIM for a given source directory.
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
  if(NOT TARGET roc::rocprim)
    add_library(roc::rocprim ALIAS rocprim)
  endif()
  if(NOT TARGET roc::rocprim_hip)
    add_library(roc::rocprim_hip ALIAS rocprim_hip)
  endif()
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
  if(NOT TARGET roc::rocrand)
    add_library(roc::rocrand ALIAS rocrand)
  endif()
endmacro()

# TODO: once we hit CMake >= 3.24 change macro to function. This is because
# CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL to promote imported
# targets of find_package to the global scope.
macro(fetch_monorepo_dep)
  set(args PACKAGE VERSION)
  cmake_parse_arguments(arg_fetch "" "${args}" "" ${ARGN})

  string(TOUPPER "FETCH_${arg_fetch_PACKAGE}_METHOD" fetch_method_name)
  set(
    ${fetch_method_name} "PACKAGE"
    CACHE STRING
    "The method used to fetch ${arg_fetch_PACKAGE}"
  )
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
    set(repo_root_dir "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    set(dep_source_dir "${repo_root_dir}/projects/${arg_fetch_PACKAGE}")

    if(NOT IS_DIRECTORY "${dep_source_dir}")
      # The source directory of the dependency does not exist in the monorepo.
      # Maybe this is a spare checkout and we need to add it.
      #
      # This requires git!
      find_git()

      # Add the project directory to the spare checkout.
      execute_process(
        COMMAND
        ${GIT_PATH} "sparse-checkout" "add" "projects/${arg_fetch_PACKAGE}"
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
      "Unknown fetch method in 'fetch_monorepo_dep': ${${fetch_method_name}}"
    )
  endif()
endmacro()
