# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)
include(ProcessorCount)

findandcheckclangtidy()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Sets up clang-tidy command variables with appropriate compiler flags for C++ and HIP files
function(setClangTidyVars)
    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE} -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -p
                           ${CMAKE_BINARY_DIR} PARENT_SCOPE
    )
    if(NOT CLANG_TIDY_HIP_ARGS)
        message(VERBOSE "Detecting HIP include directory for clang-tidy...")
        if(HIP_INCLUDE_DIR)
            set(CLANG_TIDY_HIP_ARGS -extra-arg=-D__HIP_PLATFORM_AMD__ -extra-arg=-D__HIPCC__
                                    -extra-arg=-isystem -extra-arg=${HIP_INCLUDE_DIR}
                CACHE INTERNAL "Clang-tidy extra arguments for HIP files"
            )
            message(
                STATUS
                    "Configured clang-tidy HIP arguments with include directory: ${HIP_INCLUDE_DIR}"
            )
        else()
            message(
                WARNING
                    "Could not determine HIP include directory. Tidy checks for HIP files may fail."
            )
        endif()
    endif()
    set(CLANG_TIDY_HIP_ARGS ${CLANG_TIDY_HIP_ARGS} PARENT_SCOPE)
endfunction()

# Add the 'tidy' target to the project to run tidy on all files in the hipDNN folder.
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

        # Use prefixed target names in superbuild to avoid collisions
        if(ROCM_LIBS_SUPERBUILD)
            set(_TIDY_TARGET ${PROJECT_NAME}_tidy)
            set(_TIDY_CXX_TARGET ${PROJECT_NAME}_tidy-cxx)
        else()
            set(_TIDY_TARGET tidy)
            set(_TIDY_CXX_TARGET tidy-cxx)
        endif()

        # Target for running tidy on all files using HIP args for all files.
        add_custom_target(
            ${_TIDY_TARGET}
            COMMAND
                ${RUN_CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
                -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -source-filter "^(?!.*_deps/).*" -quiet
                -j ${CLANG_TIDY_JOBS} ${CLANG_TIDY_HIP_ARGS}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT
                "Running clang-tidy on ${PROJECT_NAME} source files (${CLANG_TIDY_JOBS} parallel jobs)..."
            VERBATIM
        )

        # Target for running tidy on all C++ language files (no HIP args)
        add_custom_target(
            ${_TIDY_CXX_TARGET}
            COMMAND
                ${RUN_CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR}
                -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -source-filter
                "^(?!.*(_deps/)).*" -quiet -j ${CLANG_TIDY_JOBS}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Running clang-tidy on ${PROJECT_NAME} C++ files (${CLANG_TIDY_JOBS} parallel jobs)..."
            VERBATIM
        )

        # Alias targets with consistent hyphenated naming
        add_custom_target(
            ${PROJECT_NAME}-tidy
            DEPENDS ${_TIDY_TARGET}
            COMMENT "Alias for ${_TIDY_TARGET}"
        )
        add_custom_target(
            ${PROJECT_NAME}-tidy-cxx
            DEPENDS ${_TIDY_CXX_TARGET}
            COMMENT "Alias for ${_TIDY_CXX_TARGET}"
        )
    else()
        message(${_not_found_log_level}
                "run-clang-tidy-20 not found. The 'tidy' targets will not be available."
        )
    endif()
endfunction()

# Enable clang-tidy checks for a specific target during compilation.
#
# @param TARGET target to enable clang-tidy checks for
function(clang_tidy_check TARGET)
    setclangtidyvars()
    if(ENABLE_CLANG_TIDY)
        set_target_properties(${TARGET} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
        if(CLANG_TIDY_HIP_ARGS)
            set(CLANG_TIDY_HIP_COMMAND ${CLANG_TIDY_COMMAND} ${CLANG_TIDY_HIP_ARGS})
            set_target_properties(${TARGET} PROPERTIES HIP_CLANG_TIDY "${CLANG_TIDY_HIP_COMMAND}")
        endif()
        set_target_properties(${TARGET} PROPERTIES C_CLANG_TIDY "${CLANG_TIDY_COMMAND}")
    endif()
endfunction()
