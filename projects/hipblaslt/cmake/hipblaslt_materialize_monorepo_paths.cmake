# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Materialize paths omitted by a monorepo sparse checkout into the build tree.
#
# This is a source-tree fallback for development and CI checkouts. Installed
# packages remain the preferred way for an independently distributed hipBLASLt
# source tree to consume shared components.
function(
    _hipblaslt_git_context
    output_git_variable
    output_root_variable
    output_revision_variable
)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        return()
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
            rev-parse --show-toplevel
        RESULT_VARIABLE _repository_result
        OUTPUT_VARIABLE _repository_root
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT _repository_result EQUAL 0)
        return()
    endif()

    execute_process(
        COMMAND
            "${GIT_EXECUTABLE}" -C "${_repository_root}" rev-parse HEAD
        RESULT_VARIABLE _revision_result
        OUTPUT_VARIABLE _repository_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT _revision_result EQUAL 0)
        return()
    endif()

    set(${output_git_variable} "${GIT_EXECUTABLE}" PARENT_SCOPE)
    set(${output_root_variable} "${_repository_root}" PARENT_SCOPE)
    set(${output_revision_variable} "${_repository_revision}" PARENT_SCOPE)
endfunction()

# Extract requested paths from the current monorepo revision.
function(hipblaslt_materialize_monorepo_paths output_root_variable)
    set(_requested_paths ${ARGN})
    if(NOT _requested_paths)
        set(${output_root_variable} "" PARENT_SCOPE)
        return()
    endif()

    _hipblaslt_git_context(
        _git_executable
        _repository_root
        _repository_revision
    )
    if(NOT _git_executable)
        set(${output_root_variable} "" PARENT_SCOPE)
        return()
    endif()

    set(_materialized_root
        "${CMAKE_CURRENT_BINARY_DIR}/rocm-libraries-shared-source"
    )
    set(_revision_file "${_materialized_root}/.source-revision")

    set(_reuse_materialized_source FALSE)
    if(EXISTS "${_revision_file}")
        file(READ "${_revision_file}" _materialized_revision)
        string(STRIP "${_materialized_revision}" _materialized_revision)
        if("${_materialized_revision}" STREQUAL "${_repository_revision}")
            set(_reuse_materialized_source TRUE)
            foreach(_requested_path IN LISTS _requested_paths)
                if(NOT EXISTS "${_materialized_root}/${_requested_path}")
                    set(_reuse_materialized_source FALSE)
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(NOT _reuse_materialized_source)
        file(REMOVE_RECURSE "${_materialized_root}")
        file(MAKE_DIRECTORY "${_materialized_root}")
        set(_materialized_archive
            "${CMAKE_CURRENT_BINARY_DIR}/rocm-libraries-shared-source.tar"
        )

        execute_process(
            COMMAND
                "${_git_executable}" -C "${_repository_root}"
                archive --format=tar
                --output "${_materialized_archive}"
                "${_repository_revision}" --
                ${_requested_paths}
            RESULT_VARIABLE _archive_result
            OUTPUT_QUIET
            ERROR_VARIABLE _materialization_error
        )
        if(_archive_result EQUAL 0)
            execute_process(
                COMMAND
                    "${CMAKE_COMMAND}" -E tar xvf
                    "${_materialized_archive}"
                WORKING_DIRECTORY "${_materialized_root}"
                RESULT_VARIABLE _extraction_result
                OUTPUT_QUIET
                ERROR_VARIABLE _extraction_error
            )
        else()
            set(_extraction_result 1)
        endif()
        file(REMOVE "${_materialized_archive}")

        if(NOT _archive_result EQUAL 0 OR NOT _extraction_result EQUAL 0)
            file(REMOVE_RECURSE "${_materialized_root}")
            message(STATUS
                "Could not materialize shared monorepo source from Git: "
                "${_materialization_error}${_extraction_error}"
            )
            set(${output_root_variable} "" PARENT_SCOPE)
            return()
        endif()

        file(WRITE "${_revision_file}" "${_repository_revision}\n")
        message(STATUS
            "Materialized sparse-checkout dependencies from Git revision "
            "${_repository_revision}"
        )
    endif()

    set(${output_root_variable} "${_materialized_root}" PARENT_SCOPE)
endfunction()
