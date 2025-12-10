# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(GIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Search upward to find .git folder
while(NOT EXISTS "${GIT_ROOT_DIR}/.git" AND NOT "${GIT_ROOT_DIR}" STREQUAL "/")
    get_filename_component(GIT_ROOT_DIR "${GIT_ROOT_DIR}" DIRECTORY) # Update to parent dir
endwhile()

if(EXISTS "${GIT_ROOT_DIR}/.git")
    set(GIT_DIR "${GIT_ROOT_DIR}/.git")

    if(NOT IS_DIRECTORY "${GIT_DIR}")
        # .git is a file (ie. submodule or worktree), read the actual git dir path
        file(READ "${GIT_DIR}" GIT_DIR_CONTENT)

        # Capture the gitdir value
        string(REGEX MATCH "gitdir: ([^\n\r]+/\.git)" _ "${GIT_DIR_CONTENT}")
        if(CMAKE_MATCH_1)
            set(GIT_DIR "${CMAKE_MATCH_1}")
            if(NOT IS_ABSOLUTE "${GIT_DIR}")
                get_filename_component(GIT_DIR "${GIT_ROOT_DIR}/${GIT_DIR}" ABSOLUTE)
            endif()
        endif()
    endif()

    find_program(PRE_COMMIT_EXECUTABLE pre-commit)

    if(PRE_COMMIT_EXECUTABLE)
        if(NOT EXISTS "${GIT_DIR}/hooks/pre-commit")
            message(WARNING "\n"
                            "Pre-commit hooks are NOT installed at ${GIT_DIR}/hooks/pre-commit\n"
                            "Please run:\n" "  cd ${GIT_ROOT_DIR}\n" "  pre-commit install\n"
            )
        else()
            message(STATUS "Found git pre-commit hook installed at ${GIT_DIR}/hooks/pre-commit")
        endif()
    else()
        message(WARNING "\n" "pre-commit package is NOT installed\n" "Please run:"
                        "  pip install pre-commit\n" "  cd ${GIT_ROOT_DIR}\n"
                        "  pre-commit install\n"
        )
    endif()
endif()
