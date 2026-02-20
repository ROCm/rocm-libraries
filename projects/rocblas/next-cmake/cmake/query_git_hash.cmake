# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(query_git_hash project_name working_path)
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${working_path}
        OUTPUT_VARIABLE ${project_name}_VERSION_COMMIT_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(${project_name}_VERSION_COMMIT_ID "${${project_name}_VERSION_COMMIT_ID}" CACHE INTERNAL "Git commit ID")
endfunction()
