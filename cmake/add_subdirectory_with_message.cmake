# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Adds a subdirectory to the build with a status message, and optionally checks for an expected target.
#
# Usage:
#   add_subdirectory_with_message(<subdir> [EXPECT_TARGET <target>])
#
# Arguments:
#   <subdir>               - Relative path to the subdirectory to add.
#   EXPECT_TARGET <target> - Target name expected to be defined by the subdirectory.
function(add_subdirectory_with_message _subdir)
    cmake_parse_arguments(ARG "" "EXPECT_TARGET" "" ${ARGN})

    message(STATUS "[rocm-libraries] Configuring ${_subdir}")

    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/${_subdir}")

    if(ARG_EXPECT_TARGET AND NOT TARGET ${ARG_EXPECT_TARGET})
        message(
            FATAL_ERROR
                "[rocm-libraries] Expected target ${ARG_EXPECT_TARGET} not found in ${_subdir}"
        )
    endif()
endfunction()