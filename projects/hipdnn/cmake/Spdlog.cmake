# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Shared spdlog/fmt configuration for hipDNN components
# This module provides a unified function to enable spdlog support for any target.

# Function to enable spdlog support for a target
# This handles both system-installed spdlog and locally fetched spdlog,
# and properly configures fmt (bundled or external).
#
# Usage: hipdnn_enable_spdlog(TARGET_NAME)
#
# This function:
# - Finds spdlog if not already available
# - Adds spdlog include directories (header-only, no linking to avoid compile option inheritance)
# - Configures fmt (external or bundled)
# - Adds required compile definitions (FMT_HEADER_ONLY, etc.)
#
# Note: We use hipdnn_add_dependency_includes() instead of target_link_libraries() to avoid
# inheriting compile options like /Zc:__cplusplus from spdlog which are incompatible with
# clang++ on Windows.
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

    # Determine which spdlog target to use
    if(TARGET spdlog::spdlog_header_only)
        set(_spdlog_target spdlog::spdlog_header_only)
    elseif(TARGET spdlog_header_only)
        set(_spdlog_target spdlog_header_only)
    else()
        message(FATAL_ERROR "hipdnn_enable_spdlog: spdlog not found. "
            "Ensure spdlog is installed or available via CMAKE_PREFIX_PATH.")
    endif()

    # Check if spdlog was built with external fmt by inspecting its compile definitions
    get_target_property(_spdlog_defs ${_spdlog_target} INTERFACE_COMPILE_DEFINITIONS)
    set(_spdlog_uses_external_fmt FALSE)
    if(_spdlog_defs)
        if("SPDLOG_FMT_EXTERNAL" IN_LIST _spdlog_defs)
            set(_spdlog_uses_external_fmt TRUE)
        endif()
    endif()

    # Use include-only approach to avoid inheriting compile options (e.g., /Zc:__cplusplus)
    # that are incompatible with clang++ on Windows
    hipdnn_add_dependency_includes(${TARGET_NAME} ${_spdlog_target}
        COMPILE_DEFINITIONS FMT_HEADER_ONLY)

    # Handle external fmt configuration
    # Only add SPDLOG_FMT_EXTERNAL if spdlog was actually built with it.
    if(_spdlog_uses_external_fmt)
        find_package(fmt QUIET)
        if(NOT fmt_FOUND)
            message(FATAL_ERROR "hipdnn_enable_spdlog: spdlog requires external fmt but fmt was not found. "
                "Ensure fmt is installed or available via CMAKE_PREFIX_PATH.")
        endif()
        if(TARGET fmt::fmt-header-only)
            hipdnn_add_dependency_includes(${TARGET_NAME} fmt::fmt-header-only
                COMPILE_DEFINITIONS SPDLOG_FMT_EXTERNAL)
        elseif(TARGET fmt::fmt)
            hipdnn_add_dependency_includes(${TARGET_NAME} fmt::fmt
                COMPILE_DEFINITIONS SPDLOG_FMT_EXTERNAL)
            target_link_libraries(${TARGET_NAME} PUBLIC fmt::fmt)
        else()
            message(FATAL_ERROR "hipdnn_enable_spdlog: fmt package found but no usable target. "
                "Expected fmt::fmt-header-only or fmt::fmt.")
        endif()
    endif()

    message(STATUS "hipdnn_enable_spdlog: Enabled spdlog for target ${TARGET_NAME}")
endfunction()
