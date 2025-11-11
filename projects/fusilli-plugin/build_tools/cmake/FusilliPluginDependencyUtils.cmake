# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#===------------------------------------------------------------------------===#
#
# Provides correctly configured dependencies for fusilli-plugin build.
#
# Main entry point:
#   fusilli_plugin_dependency(DEP_NAME [args...])
#
# `fusilli_plugin_dependency` routes to lower level `_fetch_X` macros to
# actually fetch dependency `X` using FetchContent.
#
# Supported dependencies: GTest
#
#===------------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.25.2)

include(FetchContent)

# Provide a fusilli plugin dependency. `fusilli_plugin_dependency` will
# fetch the dependency using FetchContent for dependencies not available
# as system packages.
#
#  fusilli_plugin_dependency(
#    DEP_NAME
#    [<dependency-specific args>...]
#  )
#
# DEP_NAME
#   Supported dependencies:
#     GTest - Fetched via FetchContent (not provided by TheRock)
#
# <dependency-specific args>
#   The `_fetch_X` macro for dependency X defines the available options.
#   Example: GTEST_VERSION for GTest
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
        FIND_PACKAGE_ARGS # When FIND_PACKAGE_ARGS is passed CMake first tries find_package before downloading.
    )
    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK OFF)
    FetchContent_MakeAvailable(GTest)
endmacro()
