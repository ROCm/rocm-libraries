# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)
include(ProcessorCount)

findandcheckclangtidy()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# clang_tidy_check_override_args(<out-var> <clang-tidy-major-version>)
#
#   Build the single `-checks=` argument that adapts the shared .clang-tidy to one
#   particular clang-tidy binary, and set <out-var> to it; <out-var> is an empty list when
#   nothing needs adapting. `-checks=` extends `-config-file=` instead of replacing it, so
#   everything the config enables stays enabled apart from what is subtracted here.
#
#   Composed into one argument on purpose: clang-tidy keeps only the last `-checks=` it is
#   given, so a second one would silently discard the first.
#
#   Two clang-tidy binaries share .clang-tidy and they are different LLVM releases: the
#   image's (host C++, EXPECTED_CLANG_TIDY_VERSION) and ROCm's (embedded HIP kernels,
#   EXPECTED_ROCM_CLANG_TIDY_VERSION; see hip-kernel-provider's KernelTidy.cmake). Rather
#   than reduce the config to the checks both understand -- which would cost the older one
#   coverage -- the config lists the union and this table subtracts per binary. clang-tidy
#   ignores check names it does not know, so only names that are known-but-wrong for a
#   given version need to appear below.
function(clang_tidy_check_override_args out_var tidy_major_version)
    set(_disabled_checks "")

    # Windows-only relaxations: both checks fire on Microsoft STL internals rather than on
    # provider code, and are clean against libstdc++, so Linux keeps the full rule set. MSVC
    # rethrows with a bare, unnamed `throw;` (bugprone-exception-escape, which
    # IgnoredExceptions cannot narrow) and its node-based containers are not
    # nothrow-move-constructible (performance-noexcept-move-constructor).
    if(WIN32)
        list(APPEND _disabled_checks bugprone-exception-escape
             performance-noexcept-move-constructor
        )
    endif()

    # LLVM 23 renamed performance-faster-string-find to
    # performance-prefer-single-char-overloads and kept the old name as a deprecated alias
    # that emits a configuration diagnostic when it is enabled on its own. .clang-tidy
    # enables both names so that LLVM 20, which only knows the old one, keeps the check.
    # Subtracting the alias here leaves the canonical name doing the work on LLVM 23, which
    # both silences the deprecation diagnostic and avoids the duplicate warnings that
    # having both names enabled would otherwise produce.
    if(tidy_major_version MATCHES "^[0-9]+$")
        if(tidy_major_version GREATER_EQUAL 23)
            list(APPEND _disabled_checks performance-faster-string-find)
        endif()
    endif()

    if(NOT _disabled_checks)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    list(TRANSFORM _disabled_checks PREPEND "-")
    list(JOIN _disabled_checks "," _checks_value)
    # One dash: clang-tidy accepts both spellings, run-clang-tidy's argument parser only
    # understands `-checks`.
    set(${out_var} "-checks=${_checks_value}" PARENT_SCOPE)
endfunction()

# Sets up clang-tidy command variables with appropriate compiler flags for C++ and HIP files
function(setClangTidyVars)
    clang_tidy_check_override_args(_check_override_args "${CLANG_TIDY_EXE_MAJOR_VERSION}")
    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE} -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -p
                           ${CMAKE_BINARY_DIR} ${_check_override_args} PARENT_SCOPE
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
        clang_tidy_check_override_args(_check_override_args "${CLANG_TIDY_EXE_MAJOR_VERSION}")

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
                -config-file=${PROJECT_SOURCE_DIR}/.clang-tidy -source-filter "^(?!.*_deps/).*" ${_check_override_args} -quiet
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
                "^(?!.*(_deps/)).*" ${_check_override_args} -quiet -j ${CLANG_TIDY_JOBS}
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
