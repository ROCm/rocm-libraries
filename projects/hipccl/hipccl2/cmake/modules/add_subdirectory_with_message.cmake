# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Adds a subdirectory to the build with a status message,
# and optionally checks for an expected target.
#
# Usage:
#   add_subdirectory_with_message(COMPONENT <component> PREFIX_PATH <prefix> [EXPECT_TARGET <target>])
#
# Arguments:
#   COMPONENT <component>  - Component name (e.g., "mxdatagenerator").
#   PREFIX_PATH <prefix>   - Path prefix (e.g., "shared", "projects").
#   EXPECT_TARGET <target> - Optional target name produced by the subdirectory.
function(add_subdirectory_with_message)
    cmake_parse_arguments(ARG "" "COMPONENT;PREFIX_PATH;EXPECT_TARGET" "" ${ARGN})

    if(NOT ARG_COMPONENT)
        message(FATAL_ERROR "add_subdirectory_with_message: COMPONENT is required")
    endif()
    if(NOT ARG_PREFIX_PATH)
        message(FATAL_ERROR "add_subdirectory_with_message: PREFIX_PATH is required")
    endif()

    set(_subdir_path "${CMAKE_CURRENT_SOURCE_DIR}/${ARG_PREFIX_PATH}/${ARG_COMPONENT}")
    file(TO_CMAKE_PATH "${_subdir_path}" _subdir_path)

    list(APPEND CMAKE_MESSAGE_CONTEXT "${ARG_COMPONENT}")

    add_subdirectory("${_subdir_path}")

    if(ARG_EXPECT_TARGET AND NOT TARGET ${ARG_EXPECT_TARGET})
        message(FATAL_ERROR "Expected target ${ARG_EXPECT_TARGET} not found in ${_subdir_path}")
    endif()

    list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endfunction()
