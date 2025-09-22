# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===----------------------------------------------------------------------===#
#
# Provides correctly configured dependencies for fusilli-plugin build.
#
# Main entry point:
#   fusilli_plugin_dependency(DEP_NAME [args...])
#
# `fusilli_plugin_dependency` routes to lower level `_fetch_X` macros to
# actually fetch dependency `X`. Each `_fetch_X` macro preferentially
# `find_package`s installed/system versions of packages and falls back to
# vendoring dependencies in the build tree with `FetchContent`.
#
# Supported dependencies: GTest, hipdnn_frontend, Fusilli, IREERuntime
#
#===----------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.25.2)

include(FetchContent)

# Provide a fusilli plugin dependency. `fusilli_plugin_dependency` will
# preferentially use system version (available through `find_package`) of a
# dependency, and fall back to building local copy with `FetchContent` +
# configuration.
#
#  fusilli_plugin_dependency(
#    DEP_NAME
#    [<dependency-specific args>...]
#  )
#
# DEP_NAME
#   Supported dependencies:
#     GTest
#     hipdnn_frontend
#     Fusilli
#     IREERuntime
#
# <dependency-specific args>
#   The `_fetch_X` macro for dependency X defines the available options.
#   Examples: GTEST_VERSION for GTest, HIP_DNN_HASH for hipdnn_frontend
#
function(fusilli_plugin_dependency DEP_NAME)
    # Set indent for logging, any logs from dep "X" will be prefixed with [X].
    set(CMAKE_MESSAGE_INDENT "[${DEP_NAME}] ")

    # Route to appropriate _fetch_X macro. CMake macros aren't textual
    # expansions like C preprocessor macros, so a dynamic call (like below) to a
    # macro isn't a problem.
    # Macro vs function:
    #  - macros execute in caller's scope and arguments are textually substituted
    #  - functions create a new scope and arguments are real variables
    #  - both functions and macros are executed at runtime
    #
    # WARNING: Logging below checks variables it expects a _fetch_X macro to set
    #          in this scope, requiring that _fetch_X is a macro and not a
    #          function.
    if(COMMAND _fetch_${DEP_NAME})
        cmake_language(CALL _fetch_${DEP_NAME} ${ARGN})
    else()
        set(CMAKE_MESSAGE_INDENT "")
        message(FATAL_ERROR "Unknown dependency: ${DEP_NAME}")
    endif()

    # reset indent.
    set(CMAKE_MESSAGE_INDENT "")

    # FetchContent_MakeAvailable(DEP) creates a <dep>_POPULATED variable
    # indicating the dependency was fetched rather than found on system.
    #
    # WARNING: FetchContent_Declare(<name>)/FetchContent_MakeAvailable(<name>)
    #          can use anything for the name argument, if the _fetch_X macro
    #          doesn't use ${DEP_NAME} the <name>_POPULATED we're checking for
    #          here won't exist and the log may be misleading.
    string(TOLOWER ${DEP_NAME} DEP_NAME_LOWER)
    if (${DEP_NAME_LOWER}_POPULATED)
        message(STATUS "${DEP_NAME} dependency populated via FetchContent")
        message(STATUS "  Source: ${${DEP_NAME_LOWER}_SOURCE_DIR}")
        message(STATUS "  Build:  ${${DEP_NAME_LOWER}_BINARY_DIR}")
    else()
        message(STATUS "${DEP_NAME} dependency found on system via find_package")
        message(STATUS "  Config: ${${DEP_NAME}_DIR}")
    endif()
endfunction()

# GTest
#
# GTEST_VERSION
#   Version tag of GTest
macro(_fetch_GTest)
    cmake_parse_arguments(
        ARG              # prefix for parsed variables
        ""               # options (flags)
        "GTEST_VERSION"  # single-value arguments
        ""               # multi-value arguments
        ${ARGN}
    )
    if(NOT DEFINED ARG_GTEST_VERSION)
        message(FATAL_ERROR "GTEST_VERSION is required")
    endif()

    FetchContent_Declare(
        GTest
        URL https://github.com/google/googletest/archive/refs/tags/v${ARG_GTEST_VERSION}.zip
    )
    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK OFF)
    FetchContent_MakeAvailable(GTest)
endmacro()

# hipdnn_frontend
#
# HIP_DNN_HASH
#   Git commit hash or tag to fetch
macro(_fetch_hipdnn_frontend)
    cmake_parse_arguments(
        ARG                        # prefix for parsed variables
        ""                         # options (flags)
        "HIP_DNN_HASH;LOCAL_PATH"  # single-value arguments
        ""                         # multi-value arguments
        ${ARGN}
    )
    if(NOT DEFINED ARG_LOCAL_PATH AND NOT DEFINED ARG_HIP_DNN_HASH)
        message(FATAL_ERROR "Required argument: one of LOCAL_PATH or HIP_DNN_HASH")
    endif()

    if(DEFINED ARG_LOCAL_PATH AND DEFINED ARG_HIP_DNN_HASH)
        message(FATAL_ERROR "Argument error: passing both LOCAL_PATH and HIP_DNN_HASH is ambiguous.")
    endif()

    if (DEFINED ARG_HIP_DNN_HASH)
        FetchContent_Declare(
            hipdnn_frontend
            GIT_REPOSITORY https://github.com/ROCm/hipDNN.git
            GIT_TAG        ${ARG_HIP_DNN_HASH}
            # When FIND_PACKAGE_ARGS is passed, FetchContent_Declare tries to
            # find_package an installed version before downloading.
            FIND_PACKAGE_ARGS CONFIG
        )
    else()
        FetchContent_Declare(
            hipdnn_frontend
            SOURCE_DIR ${ARG_LOCAL_PATH}
        )
    endif()

    set(HIP_DNN_BUILD_BACKEND ON)
    set(HIP_DNN_BUILD_FRONTEND ON)
    set(HIP_DNN_SKIP_TESTS ON)
    set(HIP_DNN_BUILD_PLUGINS OFF)
    set(ENABLE_CLANG_TIDY OFF)
    # PIC required to link static library into shared object.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    FetchContent_MakeAvailable(hipdnn_frontend)
endmacro()

# IREERuntime
#
# NOTE: For now, we're not providing a FetchContent fallback for IREERuntime.
#       Fusilli expects that the system provides this dependency, and we're
#       keeping the projects in sync as much as possible for now. If you're
#       running in the fusilli docker container (described in sharkfuser README)
#       passing -DIREERuntime_DIR=/workspace/.cache/docker/iree/build/lib/cmake/IREE
#       should be enough.
macro(_fetch_IREERuntime)
    find_package(IREERuntime CONFIG REQUIRED)
endmacro()

# Fusilli
#
# USE_LOCAL
#   If set, uses local source from ../sharkfuser directory. Without USE_LOCAL,
#   requires system installation via find_package.
macro(_fetch_Fusilli)
    cmake_parse_arguments(
        ARG          # prefix for parsed variables
        ""           # options (flags)
        "USE_LOCAL"  # single-value arguments
        ""           # multi-value arguments
        ${ARGN}
    )

    if(NOT DEFINED ARG_USE_LOCAL)
        message(FATAL_ERROR "USE_LOCAL argument is required")
    endif()

    if(NOT ARG_USE_LOCAL)
        # For the time being we're keeping fusilli-plugin setup as in sync as
        # possible with fusilli.
        message(FATAL_ERROR "Only LOCAL builds are supported currently")
    endif()

    message(STATUS "Using local Fusilli build from ../sharkfuser")
    FetchContent_Declare(
        Fusilli
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../sharkfuser
    )
    set(FUSILLI_BUILD_TESTS OFF)
    FetchContent_MakeAvailable(Fusilli)
endmacro()
