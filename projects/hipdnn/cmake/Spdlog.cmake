# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared spdlog/fmt configuration for hipDNN components (backend, plugins, etc.)
# This module provides a unified function to enable spdlog support for any target.

# Function to enable spdlog support for a target
# This handles both system-installed spdlog and locally fetched spdlog,
# and properly configures fmt (bundled or external).
#
# Usage: hipdnn_enable_spdlog(TARGET_NAME)
#
# This function:
# - Finds spdlog if not already available
# - Links spdlog::spdlog_header_only (or creates alias if needed)
# - Configures fmt (external or bundled)
# - Adds required compile definitions (HIPDNN_PLUGIN_USE_SPDLOG, FMT_HEADER_ONLY, etc.)
#
function(hipdnn_enable_spdlog TARGET_NAME)
    # Try to find spdlog if not already available
    if(NOT TARGET spdlog::spdlog_header_only AND NOT TARGET spdlog_header_only)
        find_package(spdlog QUIET)
    endif()

    # Handle locally fetched spdlog (creates alias if needed)
    if(TARGET spdlog_header_only AND NOT TARGET spdlog::spdlog_header_only)
        add_library(spdlog::spdlog_header_only ALIAS spdlog_header_only)
    endif()

    # Check if spdlog target exists after find attempt
    if(NOT TARGET spdlog::spdlog_header_only)
        message(FATAL_ERROR "hipdnn_enable_spdlog: spdlog::spdlog_header_only target not found. "
            "Ensure spdlog is installed or available via CMAKE_PREFIX_PATH.")
    endif()

    # Find fmt (optional - may be bundled with spdlog or external)
    find_package(fmt QUIET)

    # Add spdlog via header-only approach
    target_link_libraries(${TARGET_NAME} PUBLIC spdlog::spdlog_header_only)

    # Enable plugin-style logging macros and header-only fmt
    target_compile_definitions(${TARGET_NAME} PUBLIC HIPDNN_PLUGIN_USE_SPDLOG FMT_HEADER_ONLY)

    # Handle external fmt configuration
    if(fmt_FOUND)
        target_compile_definitions(${TARGET_NAME} PUBLIC SPDLOG_FMT_EXTERNAL)
        if(TARGET fmt::fmt-header-only)
            target_link_libraries(${TARGET_NAME} PUBLIC fmt::fmt-header-only)
        endif()
    endif()

    message(STATUS "hipdnn_enable_spdlog: Enabled spdlog for target ${TARGET_NAME}")
endfunction()
