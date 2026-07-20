# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Reusable clang-tidy CMake module. Any project can include this to get
# a `tidy` build target and per-target CXX_CLANG_TIDY integration.
#
# Configurable variables (set BEFORE include(ClangTidy)):
#
#   ENABLE_CLANG_TIDY         - OFF by default. When ON, missing clang-tidy
#                                is a fatal error.
#   CLANG_TIDY_SOURCE_FILTER  - Regex for run-clang-tidy -source-filter.
#                                Default: exclude files under CMAKE_BINARY_DIR.
#   CLANG_TIDY_EXTRA_ARGS     - Extra args forwarded to clang-tidy invocations.
#                                e.g. HIP projects can set:
#                                  -extra-arg=-D__HIP_PLATFORM_AMD__
#                                  -extra-arg=-isystem -extra-arg=${HIP_INCLUDE_DIR}
#   CLANG_TIDY_CONFIG_FILE    - Path to .clang-tidy config file.
#                                Default: ${PROJECT_SOURCE_DIR}/.clang-tidy

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)
include(ProcessorCount)

findandcheckclangtidy()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Apply defaults for configurable variables
if(NOT DEFINED CLANG_TIDY_CONFIG_FILE)
    set(CLANG_TIDY_CONFIG_FILE "${PROJECT_SOURCE_DIR}/.clang-tidy")
endif()

if(NOT DEFINED CLANG_TIDY_SOURCE_FILTER)
    set(CLANG_TIDY_SOURCE_FILTER "^(?!${CMAKE_BINARY_DIR}/).*")
endif()

# Set the CLANG_TIDY_COMMAND variable for per-target integration.
function(setClangTidyVars)
    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE} -config-file=${CLANG_TIDY_CONFIG_FILE} -p
                           ${CMAKE_BINARY_DIR} ${CLANG_TIDY_EXTRA_ARGS} PARENT_SCOPE
    )
endfunction()

# Create a 'tidy' custom target that runs clang-tidy on all sources.
function(add_clang_tidy_custom_target)
    if(WIN32)
        message(STATUS "Skipped creating 'tidy' targets; not available on Windows")
        return()
    endif()
    if(ENABLE_CLANG_TIDY)
        set(_not_found_log_level WARNING)
    else()
        set(_not_found_log_level STATUS)
    endif()
    if(RUN_CLANG_TIDY_EXE)
        processorcount(N)
        if(NOT N EQUAL 0)
            set(CLANG_TIDY_JOBS ${N})
        else()
            set(CLANG_TIDY_JOBS 1)
        endif()

        if(ROCM_LIBS_SUPERBUILD)
            set(_TIDY_TARGET ${PROJECT_NAME}_tidy)
        else()
            set(_TIDY_TARGET tidy)
        endif()

        add_custom_target(
            ${_TIDY_TARGET}
            COMMAND
                ${RUN_CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
                -config-file=${CLANG_TIDY_CONFIG_FILE}
                -source-filter "${CLANG_TIDY_SOURCE_FILTER}"
                -quiet -j ${CLANG_TIDY_JOBS}
                ${CLANG_TIDY_EXTRA_ARGS}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT
                "Running clang-tidy on ${PROJECT_NAME} (${CLANG_TIDY_JOBS} parallel jobs)..."
            VERBATIM
        )

        add_custom_target(
            ${PROJECT_NAME}-tidy
            DEPENDS ${_TIDY_TARGET}
            COMMENT "Alias for ${_TIDY_TARGET}"
        )
    else()
        message(${_not_found_log_level}
                "run-clang-tidy not found. The 'tidy' targets will not be available."
        )
    endif()
endfunction()

# Enable per-compilation clang-tidy checking on a CMake target.
function(clang_tidy_check TARGET)
    setclangtidyvars()
    if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
        set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        set_target_properties(${TARGET} PROPERTIES C_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()
